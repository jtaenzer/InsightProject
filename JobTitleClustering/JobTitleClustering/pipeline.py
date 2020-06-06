import sys
import os
import bson
import numpy as np
import pandas as pd
from pymongo import MongoClient
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
pipeline.py extracts information from the mongoDB
"""


class Pipeline:
    def __init__(self, db_name, collection_name, client_url="mongodb://localhost:27017/", binary_path="."):
        self.client = MongoClient(client_url)
        # Consider adding some protections against bad input
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.binary_path = binary_path

        self.data_raw = list()
        self.titles_raw = list()

        self.data_clean = list()
        self.titles_clean = list()
        self.cleaning_level = {"title_ignore_list": False,
                               "title_depth": False,
                               "joined": False,
                               "count_vec_sum": False}

        self.skills_vocabulary = None
        self.token_pattern = "(?u)\\b[\\w\\s]{1,}\\b"
        self.count_vectorizer = None
        self.data_count_matrix = None
        self.tfidf_transformer = None
        self.data_tfidf_matrix = None

    # Creates a list strings where each string contains the skills from a profile in the DB
    # Used to create categorical dataset for the clustering model
    def get_all_skills(self, min_skill_length=5):
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"experience.title": {"$exists": True, "$ne": []}},
                               {"primary.job": {"$exists": True, "$ne": None}}]}
        mask = {"_id": 0, "skills": 1, "experience.is_primary": 1, "experience.title.name": 1}
        skills = []
        titles = []
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            # Make sure we get skill some skills from the profile and there are more than min_skill_length skills
            tmp_skills = []
            try:
                tmp_skills.extend([skill["name"] for skill in profile["skills"]])
            except KeyError:
                continue
            if len(tmp_skills) < min_skill_length:
                continue
            # Make sure we get a title from the profile
            try:
                titles.append(profile["experience"][0]["title"]["name"])
            except (KeyError, IndexError):
                continue
            skills.append(tmp_skills)
        if len(titles) != len(skills):
            print("Pipeline.get_all_skills: len(titles) != len(skills_data), this should never happen.")
            sys.exit(2)
        return titles, skills

    # Similar to get_all_skills but allows filtering by known job titles
    # Current job title filtering is simple: Find job titles starting with strings in the titles list using regex
    def get_skills_by_titles(self, titles, min_skill_length=5):
        if not titles:
            return
        # Build a list of regex expressions to use in the conditions dictionary for the DB query
        regx_exprs = [bson.regex.Regex("^{}".format(title)) for title in titles]
        conditions = {"skills": {"$exists": True, "$ne": []}}
        or_list = []
        for regx_expr in regx_exprs:
            or_list.append({"experience.0.title.functions": regx_expr})
        conditions["$or"] = or_list
        mask = {"_id": 0, "experience.title.name": 1, "skills": 1}

        # Collect relevant data from the query
        skills_data = []
        titles = []
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            tmp_skills = []
            # Make sure we get skill some skills from the profile and there are more than min_skill_length skills
            try:
                tmp_skills.extend([skill["name"] for skill in profile["skills"]])
            except KeyError:
                continue
            if len(tmp_skills) < min_skill_length:
                continue
            # Make sure we get a title from the profile
            try:
                titles.append(profile["experience"][0]["title"]["name"])
            except (KeyError, IndexError):
                continue
            skills_data.append(tmp_skills)
        if len(titles) != len(skills_data):
            print("Pipeline.get_skills_by_titles: len(titles) != len(skills_data), this should never happen.")
            sys.exit(2)
        return titles, skills_data

    # Similiar to get_all_skills above but the job title is retrieved from the primary field
    def get_all_skills_primary(self, min_skill_length=5):
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"primary.job": {"$exists": True, "$ne": None}},
                               {"primary.job.title.functions": {"$exists": True, "$ne": []}}]}
        mask = {"_id": 0, "skills": 1, "primary.job.title.functions": 1}
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            # Build a list of skills for the profile, continue if it is too short
            tmp_skills = list()
            try:
                tmp_skills.extend([skill["name"] for skill in profile["skills"]])
            except KeyError:
                continue
            if len(tmp_skills) < min_skill_length:
                continue
            # Loop over the job title functions map and duplicate the skills data for each title
            for title in profile["primary"]["job"]["title"]["functions"]:
                self.titles_raw.append(title)
                self.data_raw.append(tmp_skills)
            if len(self.titles_raw) != len(self.data_raw):
                print("Pipeline.get_all_skills: len(titles) != len(skills_data), this should never happen.")
                sys.exit(2)

    def prepare_data_for_count_vectorizer(self, skill_depth=10000):
        skills_flat = list()
        data_join = list()
        for row in self.data_clean:
            skills_flat.extend(row)
            data_join.append(", ".join(row))
        self.data_clean = data_join
        self.cleaning_level["joined"] = True
        skills_flat = pd.Series(skills_flat, dtype=str)
        mask = skills_flat.isin(skills_flat.value_counts()[:skill_depth].index)
        self.skills_vocabulary = skills_flat[mask].drop_duplicates().tolist()

    def setup_count_vectorizer_and_transform(self):
        self.count_vectorizer = CountVectorizer(vocabulary=self.skills_vocabulary, token_pattern=self.token_pattern)
        self.data_count_matrix = self.count_vectorizer.transform(self.data_clean)

    def setup_tfidf_transformer_and_fit_transform(self, data):
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_transformer.fit(data)
        self.data_tfidf_matrix = self.tfidf_transformer.transform(data)

    def drop_titles_from_data(self, titles_to_drop):
        if not titles_to_drop:
            self.titles_clean = pd.Series(self.titles_raw, dtype=str)
            self.data_clean = self.data_raw
            self.cleaning_level["title_ignore_list"] = True
            return
        titles_series = pd.Series(self.titles_raw, dtype=str)
        mask = ~titles_series.isin(titles_to_drop)
        self.titles_clean = titles_series[mask]
        self.data_clean = self.clean_data_list_by_mask(mask, self.data_raw)
        self.cleaning_level["title_ignore_list"] = True

    def drop_matrix_rows_by_sum(self, min_skill_length=5):
        mask = (np.sum(self.data_count_matrix, axis=1) >= min_skill_length).reshape(1, -1).tolist()[0]
        self.titles_clean = self.titles_clean[mask]
        self.data_clean = self.clean_data_list_by_mask(mask, self.data_clean)
        self.data_tfidf_matrix = self.data_tfidf_matrix.toarray()[mask]  # This may try to allocate alot of memory...
        self.cleaning_level["count_vec_sum"] = True

    def dump_binaries(self):
        if not os.path.exists(self.binary_path):
            os.makedirs(self.binary_path)
        if len(self.titles_clean) > 0:
            dump(self.titles_clean, self.binary_path + "titles_clean.joblib")
        if self.data_clean:
            dump(self.data_clean, self.binary_path + "data_clean.joblib")
        if self.skills_vocabulary:
            dump(self.skills_vocabulary, self.binary_path + "skills_vocabulary.joblib")
        if self.tfidf_transformer:
            dump(self.tfidf_transformer, self.binary_path + "tfidf_transformer.joblib")
        if self.data_tfidf_matrix.shape[0] > 0:
            dump(self.data_tfidf_matrix, self.binary_path + "data_tfidf_matrix.joblib")

    def load_binaries(self):
        self.titles_clean = load(self.binary_path + "titles_clean.joblib")
        self.data_clean = load(self.binary_path + "data_clean.joblib")
        self.skills_vocabulary = load(self.binary_path + "skills_vocabulary.joblib")
        self.tfidf_transformer = load(self.binary_path + "tfidf_transformer.joblib")
        self.data_tfidf_matrix = load(self.binary_path + "data_tfidf_matrix.joblib")

    def run_clustering_pipeline(self, min_skill_length=5, drop_titles=list(), skill_depth=10000, verbose=0):
        if verbose:
            print("Getting raw data from the DB")
        self.get_all_skills_primary(min_skill_length=min_skill_length)
        if verbose:
            print("Dropping titles from bad title list from data")
        self.drop_titles_from_data(drop_titles)
        if verbose:
            print("Preparing data for CountVectorizer and TfidfTransformer")
        self.prepare_data_for_count_vectorizer(skill_depth=skill_depth)
        if verbose:
            print("Tranforming with CountVectorizer")
        self.setup_count_vectorizer_and_transform()
        if verbose:
            print("Transforming with TfidfTransformer")
        self.setup_tfidf_transformer_and_fit_transform(self.data_count_matrix)
        if verbose:
            print("Dropping data points with too few skills")
        self.drop_matrix_rows_by_sum(min_skill_length=min_skill_length)
        if verbose:
            print("Dumping binaries")
        self.dump_binaries()
        if verbose:
            print("Pipeline complete!")

    @staticmethod
    def clean_data_list_by_mask(mask, data):
        if len(mask) != len(data):
            print("Pipeline.clean_data_list_by_mask: mask and data have inconsistent lengths, exiting.")
            sys.exit(2)
        data_masked = list()
        for index, check in enumerate(mask):
            if check:
                data_masked.append(data[index])
        return data_masked

import sys
import os
import bson
import numpy as np
import pandas as pd
from pymongo import MongoClient
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

"""
pipeline.py 
-extracts data from the mongoDB
-cleans the extracted data based on hyperparameter choices
-transforms the data with sklearn's CountVectorizer and TfidfTransformer

The full pipeline for clustering can be run via run_clustering_pipeline
"""


class Pipeline:
    def __init__(self, db_name, collection_name, client_url="mongodb://localhost:27017/", binary_path=""):
        self.client = MongoClient(client_url)
        # Consider adding some protections against bad input
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        # If this is left as the default empty string, binaries won't be dumped in run_clustering_pipline
        self.binary_path = binary_path

        # Private vars to hold the data
        self.data_raw = list()
        self.titles_raw = list()

        self.data_clean = list()
        self.titles_clean = list()
        # Used to keep track of the cleaning steps applied to the data
        self.cleaning_level = {"title_ignore_list": False,
                               "title_freq": False,
                               "joined": False,
                               "count_vec_sum": False}

        self.skills_vocabulary = None
        # Token for CountVectorizer that includes white spaces so that multi-word skills aren't ignored
        self.token_pattern = "(?u)\\b[\\w\\s]{1,}\\b"
        self.count_vectorizer = None
        self.data_count_matrix = None
        self.tfidf_transformer = None
        self.data_tfidf_matrix = None
        self.label_encoder = None
        self.titles_encoded = list()
        self.scaler = None

    # Creates a list strings where each string contains the skills from a profile in the DB
    # Deprecated in favour of get_all_skills_primary
    def get_all_skills(self, min_skill_length=5):
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"experience.title": {"$exists": True, "$ne": []}},
                               {"primary.job.title": {"$exists": True, "$ne": None}}]}
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
    def get_skills_by_titles(self, titles_to_extract, min_skill_length=5, n_profiles=0):
        if not titles_to_extract:
            return
        # Keep track of how many profiles we've saved from each title, only needed for n_profiles > 0
        counts = [0]*len(titles_to_extract)
        # Build a list of regex expressions to use in the conditions dictionary for the DB query
        regx_exprs = [bson.regex.Regex("^{}".format(title)) for title in titles_to_extract]
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"primary.job": {"$exists": True, "$ne": None}},
                               {"primary.job.title.functions": {"$exists": True, "$ne": []}}]}
        or_list = []
        for regx_expr in regx_exprs:
            or_list.append({"primary.job.title.functions.0": regx_expr})
        conditions["$or"] = or_list
        mask = {"_id": 0, "primary.job.title.functions": 1, "skills": 1}

        # Collect relevant data from the query
        skills_data = list()
        titles = list()
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            if n_profiles > 0 and np.sum(counts) >= n_profiles * len(titles_to_extract):
                break
            tmp_skills = list()
            # Make sure we get skill some skills from the profile and there are more than min_skill_length skills
            try:
                tmp_skills.extend([skill["name"] for skill in profile["skills"]])
            except KeyError:
                continue
            if len(tmp_skills) < min_skill_length:
                continue
            # Make sure we get a title from the profile
            try:
                tmp_title = profile["primary"]["job"]["title"]["functions"][0]
            except (KeyError, IndexError):
                continue
            #
            if n_profiles > 0:
                for title in titles_to_extract:
                    if title in tmp_title and counts[titles_to_extract.index(title)] < n_profiles:
                        self.titles_raw.append(title)
                        self.data_raw.append(tmp_skills)
                        counts[titles_to_extract.index(title)] += 1
                    else:
                        continue
            else:
                self.titles_raw.append(tmp_title)
                self.data_raw.append(tmp_skills)
        if len(titles) != len(skills_data):
            print("Pipeline.get_skills_by_titles: len(titles) != len(skills_data), this should never happen.")
            sys.exit(2)

    # Similar to get_all_skills above but the job title is retrieved from the primary field
    def get_all_skills_primary(self, min_skill_length=5):
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"primary.job.title": {"$exists": True, "$ne": None}},
                               {"primary.job.title.name": {"$exists": True, "$ne": None}}]}
        mask = {"_id": 0, "skills": 1, "primary.job.title.name": 1}
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
            try:
                tmp_title = profile["primary"]["job"]["title"]["name"]
            except (KeyError, IndexError):
                continue
            self.titles_raw.append(tmp_title)
            self.data_raw.append(tmp_skills)
        if len(self.titles_raw) != len(self.data_raw):
            print("Pipeline.get_all_skills: len(titles) != len(skills_data), this should never happen.")
            sys.exit(2)

    # Get all titles in the flat DB, primarily used for the title analysis script
    def get_all_titles(self, titles_to_drop):
        conditions = {}
        mask = {"_id": 0, "job_title": 1}
        profiles = self.collection.find(conditions, mask)
        titles = []
        for index, profile in enumerate(profiles):
            if index % 10000 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{:2.0f}".format(float(index / 20500000) * 100) + "%")
            title = profile["job_title"]
            if title in titles_to_drop:
                continue
            titles.append(title)
        print()
        return titles

    # Similar to get_all_titles above but for use on the version of the DB containing more information
    def get_all_titles_deepDB(self, titles_to_drop, min_skill_length=5):
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"primary.job.title": {"$exists": True, "$ne": None}}]}
        mask = {"_id": 0, "skills": 1, "primary.job.title.name": 1}
        profiles = self.collection.find(conditions, mask)
        titles = []
        for index, profile in enumerate(profiles):
            if index % 10000 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{:2.0f}".format(float(index / 3500000) * 100) + "%")
            tmp_skills = list()
            try:
                tmp_skills.extend([skill["name"] for skill in profile["skills"]])
            except KeyError:
                continue
            if len(tmp_skills) < min_skill_length:
                continue
            try:
                tmp_title = profile["primary"]["job"]["title"]["name"]
            except KeyError:
                continue
            if tmp_title in titles_to_drop:
                continue
            titles.append(tmp_title)
        print()
        return titles

    # Clean the data / titles based on
    # (1) An input list of titles to drop
    # (2) Removal of "tails" based on some minimum frequency
    # This function suffers from the confusion about the titles typing (is it a list or a series?) -- to improve
    def drop_titles_from_data(self, titles_to_drop, min_title_freq=0):
        # Start by putting the raw data/titles into the clean private vars in case we don't do any cleaning here
        # This is pretty bad since it assumes this will always be the first cleaning step -- rethink
        self.titles_clean = pd.Series(self.titles_raw, dtype=str)
        self.data_clean = self.data_raw
        # Remove "tails" in the titles from the data
        if min_title_freq > 0:
            mask = self.titles_clean.isin(self.titles_clean.value_counts()[self.titles_clean.value_counts() > min_title_freq].index)
            self.titles_clean = self.titles_clean[mask]
            self.data_clean = self.clean_data_list_by_mask(mask, self.data_clean)
            self.cleaning_level["title_freq"] = True
        # If titles to drop wasn't provided stop here
        if not titles_to_drop:
            return
        # Remove titles in the titles_to_drop list from the titles/data
        mask = ~self.titles_clean.isin(titles_to_drop)
        self.titles_clean = self.titles_clean[mask]
        self.data_clean = self.clean_data_list_by_mask(mask, self.data_clean)
        self.cleaning_level["title_ignore_list"] = True

    # In order to use CountVectorizer we need a vocabulary (not necessarily true if all skills are kept...)
    # And we need to convert our data from a list of lists to a list of strings/"documents"
    # Any skills in thee data that are not in the vocab we build here will be dropped by CountVectorizer
    def prepare_data_for_count_vectorizer(self, skill_depth=10000):
        skills_flat = list()
        data_join = list()
        # If we haven't done any cleaning previously, start from the raw data
        if not self.data_clean:
            self.titles_clean = self.titles_raw
            self.data_clean = self.data_raw
        for row in self.data_clean:
            skills_flat.extend(row)
            data_join.append(", ".join(row))
        self.data_clean = data_join
        self.cleaning_level["joined"] = True
        skills_flat = pd.Series(skills_flat, dtype=str)
        mask = skills_flat.isin(skills_flat.value_counts()[:skill_depth].index)
        self.skills_vocabulary = skills_flat[mask].drop_duplicates().tolist()

    # Initialize the CountVectorizer with our vocab and transform the data
    # self.count_vectorizer can be dumped to a binary for later use by calling dump_binaries()
    def setup_count_vectorizer_and_transform(self):
        self.count_vectorizer = CountVectorizer(vocabulary=self.skills_vocabulary, token_pattern=self.token_pattern)
        self.data_count_matrix = self.count_vectorizer.transform(self.data_clean)

    # Initialize the TfidfTransformer and fit_transform the count matrix created by setup_count_vectorizer_and_transform
    # Calling these methods out of order will not work! Add protections
    # Data is taken as an input of relying on class variables so this can be used for both encoded and string data
    def setup_tfidf_transformer_and_fit_transform(self, data):
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_transformer.fit(data)
        self.data_tfidf_matrix = self.tfidf_transformer.transform(data)

    def setup_label_encoded_and_fit_transform(self):
        self.label_encoder = LabelEncoder()
        self.titles_encoded = self.label_encoder.fit_transform(self.titles_clean)

    def setup_standard_scaler_and_fit_transform(self, training_data):
        self.scaler = StandardScaler()
        training_data_scaled = self.scaler.fit_transform(training_data)
        return training_data_scaled

    # Drop rows from our matrices and data/titles lists that don't have a row sum > min_skill_length in the count matrix
    # Likely the data was already filtered by some min_skill_length during the DB extraction
    # However, now that we've established our CountVectorizer vocabulary, we can re-apply this requirement
    def drop_matrix_rows_by_sum(self, min_skill_length=5):
        mask = (self.data_count_matrix.sum(axis=1) >= min_skill_length).reshape(1, -1).tolist()[0]
        # This try/except is only necessary because self.titles_clean isn't consistently typed... should fix
        try:
            self.titles_clean = self.titles_clean[mask]
        except TypeError:
            self.titles_clean = pd.Series(self.titles_clean, dtype=str)
            self.titles_clean = self.titles_clean[mask]
        self.data_clean = self.clean_data_list_by_mask(mask, self.data_clean)
        self.data_tfidf_matrix = self.data_tfidf_matrix.toarray()[mask]  # This may try to allocate alot of memory...
        self.cleaning_level["count_vec_sum"] = True

    # Dump class vars to binaries for later use
    # Mostly exists for cases where a partial pre-pocessing needs to be re-started from some interim step
    def dump_binaries(self):
        # This function needs serious improvement
        # Many of these if statements will throw errors if the class vars are left as None
        if not os.path.exists(self.binary_path):
            os.makedirs(self.binary_path)
        if len(self.titles_clean) > 0:
            dump(self.titles_clean, self.binary_path + "titles_clean.joblib")
        if self.data_clean:
            dump(self.data_clean, self.binary_path + "data_clean.joblib")
            dump(self.cleaning_level, self.binary_path + "cleaning_level_dict.joblib")
        if self.skills_vocabulary:
            dump(self.skills_vocabulary, self.binary_path + "skills_vocabulary.joblib")
        if self.tfidf_transformer:
            dump(self.tfidf_transformer, self.binary_path + "tfidf_transformer.joblib")
        if self.data_tfidf_matrix.shape[0] > 0:
            dump(self.data_tfidf_matrix, self.binary_path + "data_tfidf_matrix.joblib")
        if self.data_count_matrix.shape[0] > 0:
            dump(self.data_count_matrix, self.binary_path + "data_count_matrix.joblib")
        if self.label_encoder:
            dump(self.label_encoder, self.binary_path + "label_encoder.joblib")
        if len(self.titles_encoded) > 0:
            dump(self.titles_encoded, self.binary_path + "titles_encoded.joblib")
        if self.scaler:
            dump(self.scaler, self.binary_path + "scaler.joblib")

    # Load existing binaries into class vars
    # classify_mode=True loads additional binaries needed for classifying/interpreting classification model output
    def load_binaries(self, classify_mode=False):
        self.titles_clean = load(self.binary_path + "titles_clean.joblib")
        self.data_clean = load(self.binary_path + "data_clean.joblib")
        self.cleaning_level = load(self.binary_path + "cleaning_level_dict.joblib")
        self.skills_vocabulary = load(self.binary_path + "skills_vocabulary.joblib")
        self.tfidf_transformer = load(self.binary_path + "tfidf_transformer.joblib")
        self.data_tfidf_matrix = load(self.binary_path + "data_tfidf_matrix.joblib")
        self.data_count_matrix = load(self.binary_path + "data_count_matrix.joblib")
        if classify_mode:
            self.label_encoder = load(self.binary_path + "label_encoder.joblib")
            self.titles_encoded = load(self.binary_path + "titles_encoded.joblib")
            self.scaler = load(self.binary_path + "scaler.joblib")

    # Run the full clustering pipeline in order
    def run_clustering_pipeline(self, min_skill_length=5, drop_titles=list(), skill_depth=10000, min_title_freq=3, verbose=0):
        if verbose:
            print("Getting raw data from the DB")
        self.get_all_skills_primary(min_skill_length=min_skill_length)
        if verbose:
            print("Dropping titles from bad title list from data")
        self.drop_titles_from_data(drop_titles, min_title_freq=min_title_freq)
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
        if self.binary_path:
            self.dump_binaries()
        if verbose:
            print("Pipeline complete!")

    # This function exists because the data list is generally too large to be put in a pandas series/dataframe
    # Provides the same functionality as df = df[mask] but in a for loop...
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

import sys
import os
import bson
import numpy as np
import pandas as pd
from pymongo import MongoClient
from random import sample
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
        self.titles_encoded = list()

        self.data_subsampled = dict()
        self.titles_subsampled = list()

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
        self.scaler = None

    # Get all titles in the flat DB, primarily used for the title analysis script
    # Conditions and mask may have to be modified based on DB structure
    def get_all_titles(self, drop_list=[]):
        conditions = {"clean_title": {"$exists": True, "$ne": None}}
        mask = {"_id": 0, "clean_title": 1}
        profiles = self.collection.find(conditions, mask)
        titles = []
        for index, profile in enumerate(profiles):
            if any(drop in profile["clean_title"] for drop in drop_list):
                continue
            titles.append(profile["clean_title"])
        return titles

    # Get job titles and skills from the DB
    # Titles can be filtered using drop list
    # Conditions and mask may have to be modified based on DB structure
    def get_titles_and_skills_data(self, min_skill_length=5, drop_list=[]):
        conditions = {"$and": [{"skills.{}".format(min_skill_length-1): {"$exists": True}},
                               {"clean_title": {"$exists": True, "$ne": None}}]}
        mask = {"_id": 0, "skills": 1, "clean_title": 1}
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            if index > 100: break
            if any(drop in profile["clean_title"] for drop in drop_list):
                continue
            self.titles_raw.append(profile["clean_title"])
            self.data_raw.append([skill["name"] for skill in profile["skills"]])
        if len(self.titles_clean) != len(self.data_clean):
            print("Pipeline.get_titles_and_skills_data : len(titles) != len(skills_data), this should never happen.")
            sys.exit(2)

    # Similar to get_all_skills but allows filtering by known job titles
    # Current job title filtering is simple: Find job titles starting with strings in the titles list using regex
    # This should only be used for exploratory analysis
    def get_skills_by_titles(self, titles_to_extract, min_skill_length=5, n_profiles=0):
        if not titles_to_extract:
            return
        # Keep track of how many profiles we've saved from each title, only needed for n_profiles > 0
        counts = [0]*len(titles_to_extract)
        # Build a list of regex expressions to use in the conditions dictionary for the DB query
        regx_exprs = [bson.regex.Regex("^{}".format(title)) for title in titles_to_extract]
        conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                               {"primary.job.title.name": {"$exists": True, "$ne": None}},
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
        # If titles to drop wasn't provided stop here -- note this is generally being done in the DB query methods now
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
        # Flatten the data to create skills vocab, join skills lists in preparation for tokenization
        for row in self.data_clean:
            skills_flat.extend(row)
            data_join.append(", ".join(row))
        self.data_clean = data_join
        self.cleaning_level["joined"] = True
        skills_flat = pd.Series(skills_flat, dtype=str)
        # Keep the top skill_depth skills as vocab -- could consider doing this by frequency as well
        mask = skills_flat.isin(skills_flat.value_counts()[:skill_depth].index)
        self.skills_vocabulary = skills_flat[mask].drop_duplicates().tolist()

    # Drop rows from our matrices and data/titles lists that don't have a row sum > min_skill_length in the count matrix
    # Likely the data was already filtered by some min_skill_length during the DB extraction
    # However, now that we've established our CountVectorizer vocabulary, we can re-apply this requirement
    def drop_matrix_rows_by_sum(self, min_skill_length=5):
        mask = (self.data_count_matrix.sum(axis=1) >= min_skill_length).reshape(1, -1).tolist()[0]
        # This try/except is only necessary because self.titles_clean isn't consistently typed, might be a list
        try:
            self.titles_clean = self.titles_clean[mask]
        except TypeError:
            self.titles_clean = pd.Series(self.titles_clean, dtype=str)
            self.titles_clean = self.titles_clean[mask]
        self.data_clean = self.clean_data_list_by_mask(mask, self.data_clean)
        self.data_tfidf_matrix = self.data_tfidf_matrix.toarray()[mask]  # This may try to allocate alot of memory...
        self.cleaning_level["count_vec_sum"] = True

    # Sort the skills data by unique job titles and randomly subsample up to some depth
    # The functionality of drop_matrix_rows_by_sum is included in this function for convenience
    def subsample_data(self, min_skill_length=5, subsample_depth=200):
        # Make sure count_vectorizezr, tfidf_transformer were initialized
        if self.count_vectorizer is None:
            print("Pipeline.subsample_data : count_vectorizer object wasn't initialized, exiting.")
            sys.exit(2)
        if self.tfidf_transformer is None:
            print("Pipeline.subsample_data : tfidf_transformer object wasn't initialized, exiting.")
            sys.exit(2)
        # Make sure we encoded our titles -- this is mostly about saving on memory usage
        if len(self.titles_encoded) != len(self.titles_clean):
            print("Pipeline.subsample_data : titles_encoded and titles clean have inconsistent lengths, exiting ")
            sys.exit(2)

        # If we haven't done any cleaning previously, start from the raw data
        if not self.data_clean:
            self.titles_clean = self.titles_raw
            self.data_clean = self.data_raw
        # We're going to sort the skills by titles so we can subsample, start by preparing a dict
        unique_titles = pd.Series(self.titles_encoded, dtype=int).drop_duplicates().tolist()
        data_by_title = dict()
        for title in unique_titles:
            data_by_title[title] = list()
        # Loop over our data and sort by title
        for index, row in enumerate(self.data_clean):
            data_by_title[self.titles_encoded[index]].append(row)

        # Randomly subsample the data split by titles and TF-IDF transform it
        # The functionality of drop_matrix_rows_by_sum is also applied here
        data_subsampled_matrices = dict()
        for index, key in enumerate(data_by_title):
            # Only subsample if we have more than subsample_depth profiles for a particular title
            if len(data_by_title[key]) > subsample_depth:
                self.data_subsampled[key] = sample(data_by_title[key], subsample_depth)
            else:
                self.data_subsampled[key] = data_by_title[key]
            matrix = self.count_vectorizer.transform(self.data_subsampled[key]).toarray()
            mask = np.sum(matrix, axis=1) > min_skill_length
            matrix = matrix[mask]
            # This can result in an empty matrix which will break the tfidf transformer
            if matrix.shape[0] == 0:
                continue
            self.data_subsampled[key] = self.clean_data_list_by_mask(mask, self.data_subsampled[key])
            self.cleaning_level["count_vec_sum"] = True
            matrix = self.tfidf_transformer.transform(matrix).toarray()
            # Add the title as a column so we can split it off later and have consistent indices
            titles_col = np.array([[key] * matrix.shape[0]]).reshape(-1, 1)
            data_subsampled_matrices[key] = np.concatenate((matrix, titles_col), axis=1)
        # **WARNING** This concatenation can eat up a lot of memory!
        self.data_tfidf_matrix = np.concatenate([data_subsampled_matrices[key] for key in data_subsampled_matrices.keys()], axis=0)
        # We stored the titles in the last column of the matrix so we could grab them here
        self.titles_subsampled = self.data_tfidf_matrix[:,-1]
        # Can now drop the last column from the larger matrix
        self.data_tfidf_matrix = self.data_tfidf_matrix[:, :-1]

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

    # Initialize the label encoder and transform the titles
    # This can be used to reduce memory intensity (storing titles as ints instead of strings)
    def setup_label_encoder_and_fit_transform(self):
        self.label_encoder = LabelEncoder()
        self.titles_encoded = self.label_encoder.fit_transform(self.titles_clean)

    # Initialize the standard scaler and transform the data -- needs to have been binary or TFIDF encoded already
    def setup_standard_scaler_and_fit_transform(self, training_data):
        self.scaler = StandardScaler()
        training_data_scaled = self.scaler.fit_transform(training_data)
        return training_data_scaled

    # Dump class vars to binaries for later use
    # Mostly exists for cases where a partial pre-pocessing needs to be re-started from some interim step
    def dump_binaries(self):
        # This function needs serious improvement
        # Many of these if statements will throw errors if the class vars are left as None
        if not os.path.exists(self.binary_path):
            os.makedirs(self.binary_path)
        dump(self.titles_clean, self.binary_path + "titles_clean.joblib")
        dump(self.data_clean, self.binary_path + "data_clean.joblib")
        dump(self.cleaning_level, self.binary_path + "cleaning_level_dict.joblib")
        dump(self.count_vectorizer, self.binary_path + "count_vectorizer.joblib")
        dump(self.tfidf_transformer, self.binary_path + "tfidf_transformer.joblib")
        dump(self.label_encoder, self.binary_path + "label_encoder.joblib")
        dump(self.titles_encoded, self.binary_path + "titles_encoded.joblib")
        dump(self.data_subsampled, self.binary_path + "data_subsampled.joblib")
        dump(self.titles_subsampled, self.binary_path + "titles_subsampled.joblib")
        dump(self.scaler, self.binary_path + "scaler.joblib")

    # Load existing binaries into class vars
    def load_binaries(self):
        self.titles_clean = load(self.binary_path + "titles_clean.joblib")
        self.data_clean = load(self.binary_path + "data_clean.joblib")
        self.cleaning_level = load(self.binary_path + "cleaning_level_dict.joblib")
        self.count_vectorizer = load(self.binary_path + "count_vectorizer.joblib")
        self.tfidf_transformer = load(self.binary_path + "tfidf_transformer.joblib")
        self.label_encoder = load(self.binary_path + "label_encoder.joblib")
        self.titles_encoded = load(self.binary_path + "titles_encoded.joblib")
        self.data_subsampled = load(self.binary_path + "data_subsampled.joblib")
        self.titles_subsampled = load(self.binary_path + "titles_subsampled.joblib")
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

from pymongo import MongoClient
import bson
import numpy as np

"""
pipeline.py extracts information from the mongoDB
"""


class Pipeline:
    def __init__(self, db_name, collection_name, client_url="mongodb://localhost:27017/"):
        self.client = MongoClient(client_url)
        # Consider adding some protections against bad input
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # Creates a list of lists where each sub-list contains the skills from a profile in the DB
    # Used to create categorical dataset for the clustering model
    def get_skills_list_of_lists_by_titles(self, titles=[]):
        if not titles:
            return [[]]
        # Build a list of regex expressions to use in the conditions dictionary for the DB query
        regx_exprs = [bson.regex.Regex("^{}".format(title)) for title in titles]
        conditions = {"skills": {"$exists": True, "$ne": []}}
        or_list = []
        for regx_expr in regx_exprs:
            or_list.append({"experience.0.title.functions": regx_expr})
        conditions["$or"] = or_list
        mask = {"_id": 0, "experience.title.name": 1, "skills": 1}

        # Collection relevant data from the query
        # Harded-coded [0] is dangerous but the DB should be sanitized to ensure this always exists
        skills = []
        for profile in self.collection.find(conditions, mask):
            skills.append([profile["experience"][0]["title"]["name"]] + [skill["name"] for skill in profile["skills"]])

        # Create a padded numpy array so we can take advantage of some numpy functionalities
        skills_arr = np.array(skills, dtype=object)
        lens = np.array([len(row) for row in skills])
        array_mask = np.arange(lens.max()) < lens[:, None]
        skills_arr_padded = np.empty(array_mask.shape, dtype=skills_arr.dtype)
        skills_arr_padded[:] = None
        skills_arr_padded[array_mask] = np.concatenate(skills_arr)

        return skills_arr_padded, skills

    def get_all_skills(self):
        conditions = {"skills": {"$exists": True, "$ne": []}}
        mask = {"_id": 0, "experience.title.name": 1, "skills": 1}
        skills = []
        for profile in self.collection.find(conditions, mask):
            skills.append([profile["experience"][0]["title"]["name"]] + [skill["name"] for skill in profile["skills"]])

        # Create a padded numpy array so we can take advantage of some numpy functionalities
        skills_arr = np.array(skills, dtype=object)
        lens = np.array([len(row) for row in skills])
        array_mask = np.arange(lens.max()) < lens[:, None]
        skills_arr_padded = np.empty(array_mask.shape, dtype=skills_arr.dtype)
        skills_arr_padded[:] = np.nan
        skills_arr_padded[array_mask] = np.concatenate(skills_arr)

        return skills_arr_padded, skills
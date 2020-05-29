from pymongo import MongoClient
import bson
import pandas as pd

"""
pipeline.py extracts information from the mongoDB
"""


class Pipeline:
    def __init__(self, db_name, collection_name, client_url="mongodb://localhost:27017/"):
        self.client = MongoClient(client_url)
        # Consider adding some protections against bad input
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # Returns a list of all skills in the DB associated with a given job title
    def get_skills_list_by_title(self, title):
        regx_expr = bson.regex.Regex("^{}".format(title))
        conditions = {"skills": {"$exists": True, "$ne": []}, "experience.0.title.functions": regx_expr}
        mask = {"_id": 0, "skills": 1}
        skills = []
        for profile in self.collection.find(conditions, mask):
            skills.extend([skill["name"] for skill in profile["skills"]])
        return skills

    # Returns a list of all skills in the DB
    # EXPENSIVE -- used for tokenizing the skills
    def get_all_skills_list(self):
        conditions = {"skills": {"$exists": True, "$ne": []}}
        mask = {"_id": 0, "skills": 1}
        skills = []
        for profile in self.collection.find(conditions, mask):
            skills.extend([skill["name"] for skill in profile["skills"]])
        return skills

    # Creates a list of lists where each sub-list contains the skills from a profile in the DB
    # Used to create categorical dataset for the clustering model
    def get_skills_list_of_lists_by_titles(self, titles=[]):
        if not titles:
            return [[]]
        regx_exprs = [bson.regex.Regex("^{}".format(title)) for title in titles]
        conditions = {"skills": {"$exists": True, "$ne": []}}
        or_list = []
        for regx_expr in regx_exprs:
            or_list.append({"experience.0.title.functions": regx_expr})
        conditions["$or"] = or_list
        mask = {"_id": 0, "skills": 1}
        skills = []
        for profile in self.collection.find(conditions, mask):
            skills.append([skill["name"] for skill in profile["skills"]])
        return skills

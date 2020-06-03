import sys
import bson
from pymongo import MongoClient

"""
pipeline.py extracts information from the mongoDB
"""


class Pipeline:
    def __init__(self, db_name, collection_name, client_url="mongodb://localhost:27017/"):
        self.client = MongoClient(client_url)
        # Consider adding some protections against bad input
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # Creates a list strings where each string contains the skills from a profile in the DB
    # Used to create categorical dataset for the clustering model
    def get_all_skills(self, min_skill_length=5):
        conditions = dict()
        conditions["$and"] = [{"skills": {"$exists": True, "$ne": []}},
                             {"experience.title": {"$exists": True, "$ne": []}}]
        mask = {"_id": 0, "skills": 1, "experience.is_primary": 1, "experience.title.name": 1}
        skills_flat = []
        skills_data = []
        titles = []
        profiles = self.collection.find(conditions, mask)
        for index, profile in enumerate(profiles):
            if index % 1000 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{:2.0f}".format(float(index / 3800000) * 100) + "%")
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
            skills_flat.extend(tmp_skills)
            skills_data.append(" ".join(tmp_skills))
        print()
        return titles, skills_data, skills_flat

    # Similar to get_all_skills but allows filtering by known job titles
    # Current job title filtering is simple: Find job titles starting with strings in the titles list using regex
    def get_skills_by_titles(self, titles):
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
        skills_flat = []
        titles = []
        for profile in self.collection.find(conditions, mask):
            try:
                tmp = [skill["name"] for skill in profile["skills"]]
                skills_flat.extend(tmp)
                skills_data.append(tmp)
                titles.append(profile["experience"][0]["title"]["name"])
            except (KeyError, IndexError):
                continue
        return titles, skills_data, skills_flat

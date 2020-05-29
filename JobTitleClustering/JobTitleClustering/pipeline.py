from pymongo import MongoClient
import bson
import pandas as pd

"""
pipeline.py extracts information from the mongoDB
"""


def find_profiles(table, conditions={}, mask={"_id": 0}):
    return table.find(conditions, mask)


# Connect to the MongoDB client
myclient = MongoClient("mongodb://localhost:27017/")
# Get the FutureFitAI_database from MongoClient
db = myclient["FutureFitAI_database"]
# Create a talent_profiles table
collection_profiles = db["talent_profiles"]

"""
regx = bson.regex.Regex("^data scientist")
conditions = {"skills": {"$exists": True, "$ne": []}, "experience.0.title.functions": regx}
mask = {"_id": 0, "skills": 1}
ds_skills = pd.Series(dtype=object)
for profile in find_profiles(collection_profiles, conditions, mask):
    ds_skills = ds_skills.append(pd.Series([skill["name"] for skill in profile["skills"]]))

print(ds_skills.value_counts()[ds_skills.value_counts() > 10])
print(ds_skills.value_counts()[ds_skills.value_counts() <= 10])


regx = bson.regex.Regex("^data engineer")
conditions = {"skills": {"$exists": True, "$ne": []}, "experience.0.title.functions": regx}
mask = {"_id": 0, "skills": 1}
de_skills = pd.Series(dtype=object)
for profile in find_profiles(collection_profiles, conditions, mask):
    de_skills = de_skills.append(pd.Series([skill["name"] for skill in profile["skills"]]))

print(de_skills.value_counts()[de_skills.value_counts() > 10])
print(de_skills.value_counts()[de_skills.value_counts() <= 10])
"""

regx = bson.regex.Regex("^data analyst")
conditions = {"skills": {"$exists": True, "$ne": []}, "experience.0.title.functions": regx}
mask = {"_id": 0, "skills": 1}
ai_skills = pd.Series(dtype=object)
for profile in find_profiles(collection_profiles, conditions, mask):
    ai_skills = ai_skills.append(pd.Series([skill["name"] for skill in profile["skills"]]))

print(ai_skills.value_counts()[ai_skills.value_counts() > 10])
print(ai_skills.value_counts()[ai_skills.value_counts() <= 10])
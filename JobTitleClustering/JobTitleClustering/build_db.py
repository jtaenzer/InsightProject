import json
import os
from pymongo import MongoClient
from pymongo import errors as pymongoerrors

"""
build_db.py loops over files in data/raw and adds their contents to a mongoDB

Consider moving some of the hard coded strings to a config file
"""

# Connect to the MongoDB client, create database and a table to hold talent profiles
myclient = MongoClient("mongodb://localhost:27017/")
db = myclient["FutureFitAI_database"]
collection_profiles = db["talent_profiles"]
# Create the "id" index as unique to avoid writing duplicate entries to the table
collection_profiles.create_index("id", unique=True)

data_path = "C:/Users/joe/PycharmProjects/InsightProject/data/raw/"
data_files = os.listdir(data_path)

# Loop over files, load each line as a json, and insert it to the db
for data_file in data_files:
    print("Adding file {} to the DB.".format(data_file))
    with open(data_path + data_file) as data_file_content:
        for line in data_file_content:
            try:
                collection_profiles.insert_one(json.loads(line))
            # continue if the "id" of the line exists already in the table, causing a DuplicateKeyError
            except pymongoerrors.DuplicateKeyError as err:
                continue

# Remove rows where both the skills and experience arrays are empty
# Could probably be avoided by inspecting the json files above before inserting
conditions = {"$and": [{"skills": {"$exists": True, "$eq": []}}, {"experience": {"$exists": True, "$eq": []}}]}
collection_profiles.remove(conditions)
import json
import os
from pymongo import MongoClient
from pymongo import errors as pymongoerrors

"""
build_db.py loops over files in data/raw and adds their contents to a mongoDB

Consider moving some of the hard coded strings to a config file
"""

# Connect to the MongoDB client
myclient = MongoClient("mongodb://localhost:27017/")
# Get the FutureFitAI_database from MongoClient
db = myclient["FutureFitAI_database"]
# Create a talent_profiles table
collection_profiles = db["talent_profiles"]
# Create the "id" index as unique to avoid writing duplicate entries to the table
collection_profiles.create_index("id", unique=True)

data_path = "C:/Users/joe/PycharmProjects/InsightProject/data/raw/"
data_files = os.listdir(data_path)

for data_file in data_files:
    print("Adding file {} to the DB.".format(data_file))
    with open(data_path + data_file) as data_file_content:
        for line in data_file_content:
            # Try to insert each line (converted to json) to the DB
            try:
                collection_profiles.insert_one(json.loads(line))
            # continue if the "id" of the line exists already in the table, causing a DuplicateKeyError
            except pymongoerrors.DuplicateKeyError as err:
                continue

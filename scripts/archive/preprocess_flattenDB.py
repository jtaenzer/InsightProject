import sys
import pickle
import config_preprocess as cfg
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline


def get_all_skills_flat(collection):
    conditions = dict()
    conditions["$and"] = [{"skills": {"$exists": True, "$ne": []}},
                          {"experience.title": {"$exists": True}},
                          {"experience.title": {"$ne": None}}]
    mask = {"_id": 0, "skills": 1}
    skills_flat = []
    for profile in collection.find(conditions, mask):
        try:
            skills_flat.extend([skill["name"] for skill in profile["skills"]])
        except (KeyError, IndexError):
            continue
    return skills_flat


def get_titles_and_skills(collection):
    conditions = dict()
    conditions["$and"] = [{"skills": {"$exists": True, "$ne": []}},
                          {"experience.title": {"$exists": True}},
                          {"experience.title": {"$ne": None}}]
    mask = {"_id": 0, "experience.title.name": 1, "skills": 1}
    skills_flat = []
    skills_data = []
    titles = []
    for profile in collection.find(conditions, mask):
        try:
            tmp_skills = [skill["name"] for skill in profile["skills"]]
            tmp_title = profile["experience"][0]["title"]["name"]
            if not tmp_title or len(tmp_skills) < cfg.min_skill_length:
                continue
            skills_data.append(tmp_skills)
            titles.append(tmp_title)
        except (KeyError, IndexError):
            continue
    return titles, skills_data


client = MongoClient(cfg.host)
db = client[cfg.dbname]
collection_profiles = db[cfg.collection_profiles]

# Create or load integer encoding for the cfg.skill_count_depth most frequent skill
labelenc = LabelEncoder()
if cfg.create_encoding:
    print("Preprocessing - creating encoding")
    db[cfg.collection_encoding].drop()
    #collection_encoding = db[cfg.collection_encoding]
    skills_flat = pd.Series(get_all_skills_flat(collection_profiles), dtype=str)
    mask = skills_flat.isin(skills_flat.value_counts()[:cfg.skill_count_depth].index)
    skills_flat_masked = skills_flat[mask]
    labelenc.fit(skills_flat_masked.drop_duplicates().tolist())
    #collection_encoding.insert_one({"classes": labelenc.classes_.tolist(), "fit_input": skills_flat_masked.tolist()})
    with open(cfg.encoding_path+cfg.encoding_name, "wb+") as file:
        pickle.dump(labelenc.classes_, file)
else:
    print("Preprocessing - loading encoding")
    with open(cfg.encoding_path+cfg.encoding_name, "rb") as file:
        labelenc.classes_ = pickle.load(file)

if cfg.remake_collection_flat:
    print("Preprocessing - making flat collection")
    db[cfg.collection_flat].drop()
    collection_flat = db[cfg.collection_flat]
    titles, skills = get_titles_and_skills(collection_profiles)
    if len(titles) != len(skills):
        print("Inconsistent # of titles ({0}) and # of skills lists ({1}), this should never happen."
              .format(len(titles), len(skills)))
        sys.exit(1)
    insert_list = []
    for index, row in enumerate(skills):
        # Progress indicator
        if index % 10000 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("{:2.0f}".format(float(index/len(skills))*100) + "%")

        row_enc = labelenc.transform([skill for skill in row if skill in labelenc.classes_])
        if len(row_enc) < cfg.min_skill_length:
            continue
        try:
            insert_list.append({"title": titles[index], "skills": row_enc.tolist()})
        except IndexError:
            continue
    collection_flat.insert_many(insert_list)

if cfg.remake_collection_mhot:
    collection_flat = db[cfg.collection_flat]
    insert_list = []
    profiles = collection_flat.find({}, {"_id": 1, "title": 1, "skills": 1})
    for profile in profiles:
        # Progress indicator
        if index % 10000 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("{:2.0f}".format(float(index/len(profiles))*100) + "%")
        _id = profile["_id"]
        title = profile["title"]
        skills = profile["skills"]
        row_mhot = np.zeros(shape=len(labelenc.classes_), dtype=np.uint8)
        row_mhot[skills] = 1
        collection_flat.update_one({"_id": _id}, {"$set": {"mhotvec": row_mhot.tolist()}})

import os
from pymongo import MongoClient
import re

def regex_subs(title):
    clean_title = title
    clean_title = re.sub(r"\bceo\b", "chief executive officer", clean_title)
    clean_title = re.sub(r"\bpres\b", "president", clean_title)
    clean_title = re.sub(r"\bvp\b", "vice president", clean_title)
    clean_title = re.sub(r"\bcto\b", "chief technology officer", clean_title)
    clean_title = re.sub(r"\bcfo\b", "chief financial officer", clean_title)
    clean_title = re.sub(r"\bcoo\b", "chief operating officer", clean_title)
    clean_title = re.sub(r"\bexec\b", "executive", clean_title)
    clean_title = re.sub(r"\bdir\b", "director", clean_title)
    clean_title = re.sub(r"\badmn\b", "administrative", clean_title)
    clean_title = re.sub(r"\badj\b", "adjunct", clean_title)
    clean_title = re.sub(r"\basst\b", "assistant", clean_title)
    clean_title = re.sub(r"\baud\b", "auditor", clean_title)
    clean_title = re.sub(r"\brn\b", "registered nurse", clean_title)
    clean_title = re.sub(r"\brep\b", "representative", clean_title)
    clean_title = re.sub(r"\bdept\b", "department", clean_title)
    clean_title = re.sub(r"\bdeptl\b", "departmental", clean_title)
    clean_title = re.sub(r"\bdiv\b", "division", clean_title)
    clean_title = re.sub(r"\beng\b", "engineer", clean_title)
    clean_title = re.sub(r"\bengr\b", "engineer", clean_title)
    clean_title = re.sub(r"\bgovt\b", "government", clean_title)
    clean_title = re.sub(r"\bgen\b", "general", clean_title)
    clean_title = re.sub(r"\bgrad\b", "graduate", clean_title)
    clean_title = re.sub(r"\bgrp\b", "group", clean_title)
    clean_title = re.sub(r"\bhr\b", "human resources", clean_title)
    clean_title = re.sub(r"\borg\b", "organization", clean_title)
    clean_title = re.sub(r"\binstr\b", "instructor", clean_title)
    clean_title = re.sub(r"\bfac\b", "factory", clean_title)
    clean_title = re.sub(r"\bqa\b", "quality assurance", clean_title)
    return clean_title

myclient = MongoClient("mongodb://localhost:27017/")
db = myclient["FutureFitAI_database"]
collection_profiles = db["talent_profiles"]

collection_profiles.update_many({}, {"$set": {"clean_title": None}})

conditions = {"$and": [{"skills": {"$exists": True, "$ne": []}},
                       {"experience.title": {"$exists": True, "$ne": []}},
                       {"primary.job.title.name": {"$exists": True, "$ne": None}}]}
mask = {"_id": 1, "primary.job.title": 1}

profiles = collection_profiles.find(conditions, mask)
for index, profile in enumerate(profiles):
    title = profile["primary"]["job"]["title"]["name"]
    # Could also pass re.IGNORECASE instead of using lower() here
    clean_title = title.lower()
    # Remove non alphanumeric characters
    clean_title = re.sub(r"[^\w]", " ", clean_title)
    # Remove any extra whitespace leftover from the last operation
    clean_title = re.sub(r"\s+", " ", clean_title)
    # Ignore titles containing accented characters (i.e. dropping the french)
    if re.match(r"[a-zA-Z]*[^A-Za-z \d]+[a-zA-Z]*", clean_title):
        continue
    _id = profile["_id"]
    # Abbreviation / Acronym substitution
    clean_title = regex_subs(clean_title)
    collection_profiles.update_one({"_id": _id}, {"$set": {"clean_title": clean_title}})
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
lol = data_extractor.get_skills_list_of_lists_by_titles(["data scientist", "data engineer"])

all_skills = []
for entry in lol:
    all_skills.extend(entry)

labelencoder = LabelEncoder()
all_skills_int_encode = labelencoder.fit_transform(all_skills)
all_skills_int_encode = all_skills_int_encode.reshape(len(all_skills_int_encode), 1)

lol_int_labels = [list(labelencoder.transform(entry)) for entry in lol]

labeled_df = pd.DataFrame(0, index=np.arange(len(lol)), columns=labelencoder.classes_)

for index, entry in enumerate(lol_int_labels):
    labeled_df.iloc[index, entry] = 1

Z = linkage(labeled_df.to_numpy(), method="single", metric="hamming")
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
lol = data_extractor.get_skills_list_of_lists_by_titles(["data scientist", "data engineer"])

# Flatten list of lists and encode skills as integers
all_skills = [skill for skill_list in lol for skill in skill_list]
labelencoder = LabelEncoder()
all_skills_int_encode = labelencoder.fit_transform(all_skills)
all_skills_int_encode = all_skills_int_encode.reshape(len(all_skills_int_encode), 1)
lol_int_labels = [list(labelencoder.transform(entry)) for entry in lol]

# Convert integer labeled data to a dataframe with a column for each skill
# Rows will contain 0s for skills not in the profile, 1 for skills in the profile
labeled_df = pd.DataFrame(0, index=np.arange(len(lol_int_labels)), columns=labelencoder.classes_)
for index, entry in enumerate(lol_int_labels):
    labeled_df.iloc[index, entry] = 1

# Cluster!
Z = linkage(labeled_df.to_numpy(), method="single", metric="hamming")
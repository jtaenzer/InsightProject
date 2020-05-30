import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
data = data_extractor.get_skills_list_of_lists_by_titles(["data scientist", "data engineer"])

# Flatten list of lists and encode skills as integers
data_flattened = [skill for skill_list in data for skill in skill_list]
labelencoder = LabelEncoder()
data_flattened_int_encode = labelencoder.fit_transform(data_flattened)
data_int_encode = [list(labelencoder.transform(entry)) for entry in data]

# Convert integer labeled data to a dataframe with a column for each skill
# Rows will contain 0s for skills not in the profile, 1 for skills in the profile
categorized_data_df = pd.DataFrame(0, index=np.arange(len(data_int_encode)), columns=labelencoder.classes_)
for index, entry in enumerate(data_int_encode):
    categorized_data_df.iloc[index, entry] = 1

# Cluster!
Z = linkage(categorized_data_df.to_numpy(), method="single", metric="hamming")
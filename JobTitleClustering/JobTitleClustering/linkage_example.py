import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

writeDFs = True

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
print("Getting skills")
padded_data, data = data_extractor.get_skills_list_of_lists_by_titles(["data scientist", "data engineer"])
# padded_data, data = data_extractor.get_all_skills()

print("Preprocessing data")
titles = padded_data[:, 0]
skills = padded_data[:, 1:]
skills_flat = skills[skills != None].tolist()
labelencoder = LabelEncoder()
labelencoder.fit(skills_flat)
# encode the non-padded data since our encoding doesn't know about "None" which we used to pad
# transforming row[1:] here because row[0] contains the job title which our encoding doesn't know about
skills_encoded = [labelencoder.transform(row[1:]) for row in data]
data_df_encoded = pd.DataFrame(0, index=np.arange(len(skills_encoded)), columns=labelencoder.classes_)
for index, entry in enumerate(skills_encoded):
    data_df_encoded.iloc[index, entry] = 1
data_df_encoded.insert(0, "job title", titles)

# Create a reduced dataframe by including only skills that occur in the dataset with some minimum frequency
skills_series = pd.Series(skills_flat)
mask = skills_series.isin(skills_series.value_counts()[skills_series.value_counts() > 10].index)
skills_to_include = skills_series[mask].drop_duplicates()
data_df_reduced = data_df_encoded[["job title"] + skills_to_include.tolist()]

if writeDFs:
    data_df_encoded.to_csv(r'./data_df_encoded.txt', sep=" ", index=False)
    data_df_reduced.to_csv(r'./data_df_reduced.txt', sep=" ", index=False)

print(data_df_encoded)
print(data_df_reduced)

print("Clustering")
# Cluster!
clustering = linkage(categorized_data_df.to_numpy(), method="single", metric="hamming")
with open("matrix.txt", "w+") as file:
    np.savetxt(file, clustering, fmt="%.5f")


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

writeDFs = True

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
print("Getting skills")
titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer"])
# titles, data, data_flat = data_extractor.get_all_skills()

print("Preprocessing data")
labelenc = LabelEncoder()
labelenc.fit(pd.Series(data_flat, dtype=str).drop_duplicates().tolist())
data_encoded = [labelenc.transform(row) for row in data]
data_encoded_df = pd.DataFrame(0, index=np.arange(len(data_encoded)), columns=labelenc.classes_, dtype=int)
for index, entry in enumerate(data_encoded):
    data_encoded_df.iloc[index, entry] = 1
data_encoded_df.insert(0, "job title", titles)

# Remove skills that appear with low frequency
data_flat_series = pd.Series(data_flat, dtype=str)
mask = data_flat_series.isin(data_flat_series.value_counts()[data_flat_series.value_counts() > 10].index)
cols_to_include = data_flat_series[mask].drop_duplicates()
data_reduced_df = data_encoded_df[["job title"] + cols_to_include.tolist()]

if writeDFs:
    data_encoded_df.to_csv(r'./data_df_encoded.txt', sep=" ", index=False)
    data_reduced_df.to_csv(r'./data_df_reduced.txt', sep=" ", index=False)

print("Clustering")
# Cluster!
clustering = linkage(data_reduced_df[cols_to_include.tolist()].to_numpy(), method="single", metric="hamming")
with open("matrix.txt", "w+") as file:
    np.savetxt(file, clustering, fmt="%.5f")


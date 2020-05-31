import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

writeDFs = True

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
print("Getting skills")
# titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer"])
titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer", "marketing manager", "brand manager"])
# titles, data, data_flat = data_extractor.get_all_skills()

print("Preprocessing - creating mask")
data_flat_ser = pd.Series(data_flat, dtype=str)
mask = data_flat_ser.isin(data_flat_ser.value_counts()[data_flat_ser.value_counts() > 10].index)

print("Preprocessing - reducing flat data")
data_flat_reduced = data_flat_ser[mask]

print("Preprocessing - fitting integer encoding")
labelenc = LabelEncoder()
labelenc.fit(data_flat_reduced.drop_duplicates().tolist())

print("Preprocessing - encoding data row by row")
data_encoded = []
for index, row in enumerate(data):
    sys.stdout.write("\r")
    sys.stdout.write("{:2.0f}".format(float(index/len(data))*100) + "%")
    row_enc = labelenc.transform([skill for skill in row if skill in labelenc.classes_])
    # Reject rows where none of the skills were encoded, may want to make this tune-able at some point
    if len(row_enc) < 1:
        titles.pop(index)  # Ensure len(titles) and len(data_encoded) are the same
        continue
    data_encoded.append(row_enc)
print()

# This is more pythonic but can't write progress!
# data_encoded = [labelenc.transform([skill for skill in row if skill in labelenc.classes_]) for row in data]

print("Preprocessing - creating dataframe")
data_encoded_df = pd.DataFrame(0, index=np.arange(len(data_encoded)), columns=labelenc.classes_, dtype=int)

print("Preprocessing - preparing dataframe")
for index, entry in enumerate(data_encoded):
    sys.stdout.write("\r")
    sys.stdout.write("{:2.0f}".format(float(index/len(data_encoded))*100) + "%")
    data_encoded_df.iloc[index, entry] = 1
print()
data_encoded_df.insert(0, "job title", titles)

if writeDFs:
    print("Preprocessing - writing df to file")
    data_encoded_df.to_csv(r"D:/FutureFit/data_encoded_df.txt", sep=" ", index=False)

print("Clustering")
# Cluster!
clustering = linkage(data_encoded_df[labelenc.classes_].to_numpy(), method="single", metric="hamming")
with open("D:/FutureFit/matrix.txt", "w+") as file:
    np.savetxt(file, clustering, fmt="%.5f")

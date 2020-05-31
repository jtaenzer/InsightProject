import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

writeDFs = True

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
print("Getting skills")
# titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer"])
# titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer", "marketing manager", "brand manager"])
titles, data, data_flat = data_extractor.get_all_skills()

print("Preprocessing - creating mask")
data_flat_ser = pd.Series(data_flat, dtype=str)
mask = data_flat_ser.isin(data_flat_ser.value_counts()[data_flat_ser.value_counts() > 100].index)

print("Preprocessing - reducing flat data")
data_flat_reduced = data_flat_ser[mask]

print("Preprocessing - fitting integer encoding")
labelenc = LabelEncoder()
labelenc.fit(data_flat_reduced.drop_duplicates().tolist())
with open("D:/FutureFit/encoding.pickle", "wb+") as file:
    pickle.dump(labelenc.classes_, file)

print("Preprocessing - encoding data")
# This is more pythonic but can't write progress!
# data_encoded = [labelenc.transform([skill for skill in row if skill in labelenc.classes_]) for row in data]
data_encoded = pd.DataFrame(columns=labelenc.classes_, dtype=np.uint8)
with open("D:/FutureFit/data_encoded.txt", "w+") as file:
    for index, row in enumerate(data):
        # Progress indicator
        sys.stdout.write("\r")
        sys.stdout.write("{:2.0f}".format(float(index/len(data))*100) + "%")

        row_enc = labelenc.transform([skill for skill in row if skill in labelenc.classes_])
        # Reject rows where none of the skills were encoded, may want to make this tune-able at some point
        if len(row_enc) < 1:
            titles.pop(index)  # Ensure len(titles) and len(data_encoded) are the same
            continue
        # Convert encoded row to an array of 0s and 1s corresponding to skills that were in the row
        row_mhot = np.zeros(shape=len(labelenc.classes_), dtype=np.uint8)
        row_mhot[row_enc] = 1
        file.write("{}\n".format(", ".join(str(i) for i in row_mhot)))  # Write the row to the file
        df_mhot = pd.DataFrame(data=row_mhot.reshape(1, -1), columns=labelenc.classes_, dtype=np.uint8)
        data_encoded = data_encoded.append(df_mhot, ignore_index=True)  # Append the row to our dataframe
print()

with open("D:/FutureFit/titles.pickle", "wb+") as file:
    pickle.dump(titles, file)

print("Clustering")
# Cluster!
clustering = linkage(data_encoded[labelenc.classes_].to_numpy(), method="single", metric="hamming")
with open("D:/FutureFit/matrix.txt", "w+") as file:
    np.savetxt(file, clustering, fmt="%.5f")

"""
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    clustering,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.savefig("D:/FutureFit/dendrogram.png")
plt.close()
"""
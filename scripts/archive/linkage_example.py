import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


def get_mhot_data(labelenc, titles, data):
    titles_cleaned = []
    data_encoded = pd.DataFrame(columns=labelenc.classes_, dtype=np.uint8)
    for index, row in enumerate(data):
        # Progress indicator
        if index % 1000 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("{:2.0f}".format(float(index / len(data)) * 100) + "%")

        row_enc = labelenc.transform([skill for skill in row if skill in labelenc.classes_])
        # Reject rows where none of the skills were encoded, may want to make this tune-able at some point
        if len(row_enc) < 3:
            continue
        titles_cleaned.append(titles[index])
        # Convert encoded row to an array of 0s and 1s corresponding to skills that were in the row
        row_mhot = np.zeros(shape=len(labelenc.classes_), dtype=np.uint8)
        row_mhot[row_enc] = 1
        df_mhot = pd.DataFrame(data=row_mhot.reshape(1, -1), columns=labelenc.classes_, dtype=np.uint8)
        data_encoded = data_encoded.append(df_mhot, ignore_index=True)  # Append the row to our dataframe
    print()
    return titles_cleaned, data_encoded

data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
print("Getting skills")
# titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer"])
# titles, data, data_flat = data_extractor.get_skills_by_titles(["data scientist", "data engineer", "marketing manager", "brand manager"])
# titles, data, data_flat = data_extractor.get_all_skills()

titles_ds, data_ds, data_flat_ds = data_extractor.get_skills_by_titles(["data scientist"])
#titles_de, data_de, data_flat_de = data_extractor.get_skills_by_titles(["data engineer"])
titles_mm, data_mm, data_flat_mm = data_extractor.get_skills_by_titles(["marketing manager"])
#titles_bm, data_bm, data_flat_bm = data_extractor.get_skills_by_titles(["brand manager"])


data_flat_ser_ds = pd.Series(data_flat_ds, dtype=str)
mask_ds = data_flat_ser_ds.isin(data_flat_ser_ds.value_counts()[:100].index)
data_flat_ser_mm = pd.Series(data_flat_mm, dtype=str)
mask_mm = data_flat_ser_mm.isin(data_flat_ser_ds.value_counts()[:100].index)

data_flat_reduced = data_flat_ser_ds[mask_ds]
data_flat_reduced = data_flat_reduced.append(data_flat_ser_mm[mask_mm])

"""

data_flat_ser_mm = pd.Series(data_flat_mm, dtype=str)
with open("D:/FutureFit/counts_mm.txt", "w+") as file:
    for index, value in data_flat_ser_mm.value_counts().items():
        print(index, value)
        file.write("{0}, {1}\n".format(index, str(value)))

data_flat_ser_ds = pd.Series(data_flat_ds, dtype=str)
with open("D:/FutureFit/counts_ds.txt", "w+") as file:
    for index, value in data_flat_ser_ds.value_counts().items():
        print(index, value)
        file.write("{0}, {1}\n".format(index, str(value)))
"""

"""
print("Preprocessing - creating mask")
mask = data_flat_ser.isin(data_flat_ser.value_counts()[data_flat_ser.value_counts() > 10].index)  # Keep the 10k most frequent skills -- make this tunable

print("Preprocessing - reducing flat data")
data_flat_reduced = data_flat_ser[mask]
"""

print("Preprocessing - fitting integer encoding")
labelenc = LabelEncoder()
labelenc.fit(data_flat_reduced.drop_duplicates().tolist())
with open("D:/FutureFit/encoding.pickle", "wb+") as file:
    pickle.dump(labelenc.classes_, file)

print("Preprocessing - encoding data")
titles_clean_ds, data_encoded_ds = get_mhot_data(labelenc, titles_ds, data_ds)
titles_clean_mm, data_encoded_mm = get_mhot_data(labelenc, titles_mm, data_mm)

freq_sort_ds = data_encoded_ds.sum(axis=1).sort_values(axis=0, ascending=False).index
freq_sort_mm = data_encoded_mm.sum(axis=1).sort_values(axis=0, ascending=False).index

data_encoded_ds = data_encoded_ds.loc[freq_sort_ds]
data_encoded_mm = data_encoded_mm.loc[freq_sort_mm]

titles_clean_ds = [titles_clean_ds[i] for i in freq_sort_ds.tolist()]
titles_clean_mm = [titles_clean_mm[i] for i in freq_sort_mm.tolist()]

titles_clean = titles_clean_ds[:150]
titles_clean.extend(titles_clean_mm[:150])

data_encoded = data_encoded_ds.iloc[:150]
data_encoded = data_encoded.append(data_encoded_mm.iloc[:150])

# This is more pythonic but can't write progress!
# data_encoded = [labelenc.transform([skill for skill in row if skill in labelenc.classes_]) for row in data]
#with open("D:/FutureFit/data_encoded.txt", "w+") as file:
# file.write("{}\n".format(", ".join(str(i) for i in row_mhot)))  # Write the row to the file


with open("D:/FutureFit/titles.pickle", "wb+") as file:
    pickle.dump(titles_clean, file)

print("Clustering")
# Cluster!
clustering = linkage(data_encoded[labelenc.classes_].to_numpy(), method="weighted", metric="hamming")
with open("D:/FutureFit/matrix.txt", "w+") as file:
    np.savetxt(file, clustering, fmt="%.5f")

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    clustering,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=5.,  # font size for the x axis labels
    labels=titles_clean,
)
plt.savefig("D:/FutureFit/dendrogram.png")
plt.close()

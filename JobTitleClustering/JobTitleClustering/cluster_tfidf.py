import os
import sys
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from pipeline import Pipeline


remake_data = False
remake_transform = True
min_skill_freq = 10000
min_title_freq = 30000
min_skill_length = 5
n_cluster_stop = 1000  # Will be used to stop the model at n clusters
save_path = "./skill_freq_{0}_skill_length_{1}_title_freq_{2}/"\
    .format(min_skill_freq, min_skill_length, min_title_freq)

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(save_path):
    os.makedirs(save_path)

if remake_data:
    data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
    print("Retrieving data from DB")
    titles, data, data_flat = data_extractor.get_all_skills(min_skill_length=min_skill_length)
    if len(titles) != len(data):
        print("Length of titles inconsistent with length of data, this should never happen.")
        sys.exit(2)
    # Create vocab for TF-IDF
    vocab_for_counts = pd.Series(data_flat, dtype=str)
    # Dump data to binaries so it can be re-used later
    print("Dumping data to binaries for later use")
    dump(data, save_path + "data_join.joblib")
    dump(vocab_for_counts.drop_duplicates().tolist(), save_path + "vocab_for_counts.joblib")
    dump(titles, save_path + "titles_for_plotting.joblib")

else:
    print("Loading data from binaries")
    data = load(save_path + "data_join.joblib")
    vocab_for_counts = pd.Series(load(save_path + "vocab_for_counts.joblib"), dtype=str)
    titles = load(save_path + "titles_for_plotting.joblib")

# Keep only most frequent titles
titles_ser = pd.Series(titles, dtype=str)
mask = titles_ser.isin(titles_ser.value_counts()[:min_title_freq].index)
titles_reduced = titles_ser[mask]

data_clean = []
print("Cleaning data -- title min frequency")
for index, check in enumerate(mask):
    if index % 1000 == 0:
        sys.stdout.write("\r")
        sys.stdout.write("{:2.0f}".format(float(index / len(mask)) * 100) + "%")
    if check:
        data_clean.append(data[index])
print()

print("Transforming data to count matrix")
mask = vocab_for_counts.isin(vocab_for_counts.value_counts()[:min_skill_freq].index)
count_vectorizer = CountVectorizer(vocabulary=vocab_for_counts[mask].drop_duplicates().tolist())
data_count_matrix = count_vectorizer.transform(data_clean)
print(data_count_matrix.shape)
dump(count_vectorizer, save_path + "count_vectorizer.joblib")

print("Transforming data to tfidf matrix")
tfidf_transformer = TfidfTransformer()
data_tfidf_matrix = tfidf_transformer.fit_transform(data_count_matrix)
print(data_tfidf_matrix.shape)
dump(tfidf_transformer, save_path + "tfidf_transformer.joblib")

print("Removing empty rows from the data")
# Remove empty rows so we can use the cosine distance
data_tfidf_matrix = data_tfidf_matrix.toarray()
mask = np.sum(data_tfidf_matrix, axis=1) != 0
data_tfidf_matrix = data_tfidf_matrix[mask]
titles_reduced = np.array(titles_reduced, dtype=str)
titles_reduced = titles_reduced[mask].tolist()

print("Cleaning data -- remove empty rows")
data_clean_no_empty_rows = []
for index, check in enumerate(mask):
    if index % 1000 == 0:
        sys.stdout.write("\r")
        sys.stdout.write("{:2.0f}".format(float(index / len(mask)) * 100) + "%")
    if check:
        data_clean_no_empty_rows.append(data_clean[index])
print()

print("Clustering")
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=n_cluster_stop)
clustering = model.fit(data_tfidf_matrix)
# Dump the clustering linkage matrix, the cleaned titles/data, and the tfidf matrix
dump(clustering, save_path + "clustering.joblib")
dump(data_tfidf_matrix, save_path + "data_tfidf_matrix.joblib")
dump(titles_reduced, save_path + "titles_cleaned_for_clustering.joblib")
dump(data_clean_no_empty_rows, save_path + "data_cleaned_for_clustering.joblib")

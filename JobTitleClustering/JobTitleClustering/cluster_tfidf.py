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
skill_depth = 10000
save_path = "D:/FutureFit/clustering_tfidf_canada/skill_depth/"
min_skill_length = 5
n_cluster_stop = 1000  # Will be used to stop the model at n clusters

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
    dump(data, save_path + "data_join.joblib")
    dump(vocab_for_counts.drop_duplicates().tolist(), save_path + "vocab_for_counts.joblib")
    dump(titles, save_path + "titles_for_plotting.joblib")

else:
    data = load(save_path + "data_join.joblib")
    vocab_for_counts = pd.Series(load(save_path + "vocab_for_counts.joblib"), dtype=str)
    titles = load(save_path + "titles_for_plotting.joblib")

print("Transforming data to count matrix")
count_vectorizer = CountVectorizer(vocabulary=vocab_for_counts.drop_duplicates().tolist(), max_features=skill_depth)
data_count_matrix = count_vectorizer.transform(data)
dump(count_vectorizer, save_path + "count_vectorizer.joblib")

print("Transforming data to tfidf matrix")
tfidf_transformer = TfidfTransformer()
data_tfidf_matrix = tfidf_transformer.fit_transform(data_count_matrix)
dump(tfidf_transformer, save_path + "tfidf_transformer.joblib")

print("Removing empty rows from the data")
# Remove empty rows so we can use the cosine distance
data_tfidf_matrix = data_tfidf_matrix.toarray()
mask = np.sum(data_tfidf_matrix, axis=1) != 0
data_tfidf_matrix = data_tfidf_matrix[mask]
titles = np.array(titles, dtype=str)
titles = titles[mask].tolist()

print("Clustering")
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=n_cluster_stop)
clustering = model.fit(data_tfidf_matrix)
dump(clustering, save_path + "clustering.joblib")



import os
from joblib import dump
from sklearn.cluster import AgglomerativeClustering
import configs.cluster_config as cfg
from pipeline import Pipeline

from random import sample
import pandas as pd
import numpy as np

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(cfg.binary_path):
    os.makedirs(cfg.binary_path)

data_pipeline = Pipeline("FutureFitAI_database", "talent_profiles_CAN", binary_path=cfg.binary_path)
print("Getting titles from the DB")
data_pipeline.get_all_skills_clean_titles(min_skill_length=cfg.min_skill_length, drop_list=cfg.titles_to_drop)
print("Dropping titles from bad title list from data")
data_pipeline.drop_titles_from_data(list(), min_title_freq=cfg.min_title_freq)
print("Preparing data for CountVectorizer and TfidfTransformer")
data_pipeline.prepare_data_for_count_vectorizer(skill_depth=cfg.min_skill_depth)
print("Tranforming with CountVectorizer")
data_pipeline.setup_count_vectorizer_and_transform()
print("Transforming with TfidfTransformer")
data_pipeline.setup_tfidf_transformer_and_fit_transform(data_pipeline.data_count_matrix)
print("Integer encoding titles")
data_pipeline.setup_label_encoder_and_fit_transform()

print("Splitting data by title and subsampling")
## This should be moved into the pipeline at some point
unique_titles = pd.Series(data_pipeline.titles_encoded, dtype=int).drop_duplicates().tolist()
data_by_title_dict = dict()
for title in unique_titles:
    data_by_title_dict[title] = list()

for index, row in enumerate(data_pipeline.data_clean):
    data_by_title_dict[data_pipeline.titles_encoded[index]].append(row)

data_subsampled = dict()
data_subsampled_matrices = dict()
for index, key in enumerate(data_by_title_dict):
    data_subsampled[key] = sample(data_by_title_dict[key], cfg.subsample_depth)
    matrix = data_pipeline.count_vectorizer.transform(data_subsampled[key]).toarray()
    matrix = matrix[np.sum(matrix, axis=1) > cfg.min_skill_length]
    matrix = data_pipeline.tfidf_transformer.transform(matrix).toarray()
    titles_col = np.array([[key]*matrix.shape[0]]).reshape(-1, 1)
    data_subsampled_matrices[key] = np.concatenate((matrix, titles_col), axis=1)

data_subsampled_matrix = np.concatenate([data_subsampled_matrices[key] for key in data_subsampled_matrices.keys()], axis=0)
titles_subsampled = data_subsampled_matrix[:, -1]
dump(titles_subsampled, cfg.binary_path + "titles_subsampled.joblib")
dump(data_subsampled, cfg.binary_path + "data_subsamples.joblib")

print("Dumping binaries")
data_pipeline.dump_binaries()
print("Pipeline complete!")

print("Clustering")
print(data_subsampled_matrix.shape)
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity=cfg.affinity, linkage=cfg.linkage, n_clusters=cfg.n_cluster_stop)
clustering = model.fit(data_subsampled_matrix[:, :-1])
dump(model, cfg.binary_path + "clustering_model.joblib")

import os
from joblib import dump
from sklearn.cluster import AgglomerativeClustering
import configs.cluster_config as cfg
from pipeline import Pipeline

import pandas as pd
import numpy as np

drop_list = ["owner",
             "founder",
             "president",
             "manager",
             "partner",
             "chief executive officer",
             "director",
             "consultant",
             "retired",
             "coordinator",
             "supervisor",
             "assistant",
             "associate",
             "leader",
             "lead",
             "buyer",
             "employee",
             "self employed"
             ]

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(cfg.binary_path):
    os.makedirs(cfg.binary_path)

data_pipeline = Pipeline("FutureFitAI_database", "talent_profiles_CAN", binary_path=cfg.binary_path)
print("Getting titles from the DB")
data_pipeline.get_all_skills_clean_titles(min_skill_length=cfg.min_skill_length, drop_list=drop_list)
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
unique_titles = pd.Series(data_pipeline.titles_encoded, dtype=int).drop_duplicates().tolist()
data_by_title_dict = dict()
for title in unique_titles:
    data_by_title_dict[title] = list()

for index, row in enumerate(data_pipeline.data_clean):
    data_by_title_dict[data_pipeline.titles_encoded[index]].append(row)

data_subsampled = dict()
for index, key in enumerate(data_by_title_dict):
    matrix = data_pipeline.count_vectorizer.transform(data_by_title_dict[key]).toarray()
    matrix = matrix[np.sum(matrix, axis=1) > cfg.min_skill_length]
    matrix = np.randum.shuffle(matrix)
    matrix = data_pipeline.tfidf_transformer.transform(matrix)
    data_subsampled[key] = matrix[:cfg.subsample_depth,:]
    titles_col = np.array([[key]*data_subsampled[key].shape[0]]).reshape(-1, 1)
    data_subsampled[key] = np.concatenate((data_subsampled[key], titles_col), axis=1)

data_training_mat = np.concatenate([data_subsampled[key] for key in data_subsampled.keys()], axis=0)

print("Dumping binaries")
data_pipeline.dump_binaries()
print("Pipeline complete!")

print("Clustering")
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity=cfg.affinity, linkage=cfg.linkage, n_clusters=cfg.n_cluster_stop)
clustering = model.fit(data_pipeline.data_tfidf_matrix)
dump(model, save_path + "clustering_model.joblib")

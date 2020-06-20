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
data_pipeline.get_titles_and_skills_data(min_skill_length=cfg.min_skill_length, drop_list=cfg.titles_to_drop)
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
data_pipeline.subsample_data(min_skill_length=cfg.min_skill_length, subsample_depth=cfg.subsample_depth)
print("Dumping binaries")
data_pipeline.dump_binaries()
print("Pipeline complete!")

print("Clustering")
print(data_pipeline.data_tfidf_matrix.shape)
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity=cfg.affinity, linkage=cfg.linkage, n_clusters=cfg.n_cluster_stop)
clustering = model.fit(data_pipeline.data_tfidf_matrix)
dump(model, cfg.binary_path + "clustering_model.joblib")

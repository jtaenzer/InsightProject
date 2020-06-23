import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from pipeline import Pipeline
from analysis_tools import AnalysisTools
import configs.analysis_config as cfg

# Create AnalysisTools object for later use
tools = AnalysisTools()

# Use the pipeline for loading binaries and to store some objects
data_pipeline = Pipeline("", "", binary_path=cfg.binary_path)
data_pipeline.load_binaries()
clustering_model = load(cfg.binary_path + "clustering_model.joblib")
titles_subsampled = data_pipeline.titles_subsampled.tolist()
title_encoding = data_pipeline.label_encoder.classes_.tolist()

# Rebuild the tfidf transformed matrix
print("Rebuilding clustering input")
matrix_list = list()
for key in data_pipeline.data_subsampled.keys():
    cluster_matrix = data_pipeline.count_vectorizer.transform(data_pipeline.data_subsampled[key]).toarray()
    cluster_matrix = cluster_matrix[np.sum(cluster_matrix, axis=1) > cfg.min_skill_length]
    cluster_matrix = data_pipeline.tfidf_transformer.transform(cluster_matrix).toarray()
    matrix_list.append(cluster_matrix)
data_tfidf_matrix = np.concatenate([matrix for matrix in matrix_list], axis=0)

# Find the core skills for each title that we encoded before clustering
print("Getting core skills")
core_skills_dict = tools.build_core_skills(data_pipeline.data_subsampled, title_encoding, depth=cfg.core_skills_depth)
# Re-build the tree-like structure of the clustering so we can unwind it
print("Building clustering tree")
clustering_tree = tools.build_clustering_tree(clustering_model, titles_subsampled, title_encoding)
# Unwind the clustering tree looking for clusters with some minimum size and purity
print("Finding pure clusters")
pure_clusters = tools.find_pure_clusters(clustering_model, clustering_tree,
                                         cfg.n_target_clusters, cfg.min_clus_size, cfg.min_purity)

# Create and dump core_skills_dict
core_skills_dict = dict()
for key in data_pipeline.data_subsampled.keys():
    tmp_skills = list()
    for index, row in enumerate(data_pipeline.data_subsampled[key]):
        tmp_skills.extend(row.split(", "))
    tmp_series = pd.Series(tmp_skills, dtype=str)
    core_skills_dict[key] = tmp_series.value_counts()[:cfg.core_skills_depth].index.tolist()
dump(core_skills_dict, cfg.binary_path + "core_skills_dict.joblib")

print("Preparing training data")
# Create training data from clusters
training_data_list = list()
for cluster in pure_clusters:
    # Label
    titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
    cluster_label_enc = title_encoding.index(titles_ser.value_counts().index[0])
    # Matrix
    cluster_matrix = data_tfidf_matrix[clustering_tree[cluster]["child_indices"]]
    output_column = np.array([[cluster_label_enc]*cluster_matrix.shape[0]]).reshape(-1, 1)
    cluster_matrix = np.concatenate((cluster_matrix, output_column), axis=1)
    np.random.shuffle(cluster_matrix)
    training_data_list.append(cluster_matrix[:int(cfg.train_test_frac*cluster_matrix.shape[0]), :])

training_data_matrix = np.concatenate([matrix for matrix in training_data_list], axis=0)
np.random.shuffle(training_data_matrix)
X_train, y_train = training_data_matrix[:, :-1], training_data_matrix[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print("Fitting model")
print(X_train.shape)
mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1], int((2/3)*X_train.shape[1]),
                                        len(data_pipeline.label_encoder.classes_)), max_iter=1000, verbose=True)

mlp.fit(X_train, y_train)
dump(mlp, cfg.binary_path + "MLPClassifier_model.joblib")
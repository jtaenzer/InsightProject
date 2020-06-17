import pandas as pd
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

binary_path = "./binaries"
# How many clusters do we want to find?
n_target_clusters = 300
# Do we want to ignore small clusters?
min_clus_size = 200
# Minimum purity -- to find good quality clusters we want them to be dominated by a particular title
min_purity = 10
# Depth in skills list to be considered core skills
core_skills_depth = 10
# Train/test frac
train_test_frac = 0.8

# Load binaries
clustering = load(binary_path + "clustering_model.joblib")
titles = load(binary_path + "titles_subsampled.joblib")
titles = titles.tolist()  # Titles was saved as a pandas series
labelenc = load(binary_path + "label_encoder.joblib")
title_encoding = labelenc.classes_.tolist()  # classes_ is a numpy array but we want to make use of list.index()
data_dict = load(binary_path + "data_subsampled.joblib")
count_vectorizer = load(binary_path + "count_vectorizer.joblib")
tfidf_transformer = load(binary_path + "tfidf_transformer.joblib")

# Rebuild the tfidf transformed matrix
print("Rebuilding clustering input")
mat_list = list()
for key in data_dict.keys():
    cluster_matrix = count_vectorizer.transform(data_dict[key]).toarray()
    cluster_matrix = cluster_matrix[np.sum(cluster_matrix, axis=1) > 10]
    cluster_matrix = tfidf_transformer.transform(cluster_matrix).toarray()
    mat_list.append(cluster_matrix)
data_tfidf_matrix = np.concatenate([mat for mat in mat_list], axis=0)

print("Getting core skills")
# Create and dump core_skills_dict
core_skills_dict = dict()
for key in data_dict.keys():
    tmp_skills = list()
    for index, row in enumerate(data_dict[key]):
        tmp_skills.extend(row.split(", "))
    tmp_series = pd.Series(tmp_skills, dtype=str)
    core_skills_dict[key] = tmp_series.value_counts()[:core_skills_depth].index.tolist()
dump(core_skills_dict, binary_path + "core_skills_dict.joblib")

print("Building clustering tree")
# Generate a dictionary holding the tree-like structure of the clustering tree
# Could also be useful to have a field that holds the direct direct descendants of each cluster
children = clustering.children_
clustering_tree = dict()
for index, row in enumerate(children):
    # Check if we've found a singleton, in that case the index is exactly whats in the clustering matrix
    # Title and skills can be taken directly from the data
    if row[0] < clustering.n_leaves_:
        titles1 = [title_encoding[int(titles[int(row[0])])]]
        indices1 = [int(row[0])]
    # If we haven't found a singleton, fill indices/titles/skills from a previous iteration of this loop!
    # Note: since we're looping through the clustering matrix in order, singletons will always be added first
    else:
        titles1 = clustering_tree[int(row[0])]["child_titles"]
        indices1 = clustering_tree[int(row[0])]["child_indices"]
    # Same as above but for the other index in the clustering matrix
    if row[1] < clustering.n_leaves_:
        titles2 = [title_encoding[int(titles[int(row[1])])]]
        indices2 = [int(row[1])]
    else:
        titles2 = clustering_tree[int(row[1])]["child_titles"]
        indices2 = clustering_tree[int(row[1])]["child_indices"]

    clustering_tree[1 + index + len(children)] = {"child_titles": titles1+titles2,
                                                  "child_indices": indices1+indices2}

print("Finding pure clusters")
# Define the starting point to unwind our clustering tree -- this could be tunable
# For now we are starting from the top-most cluster which should contain all possible children
clusters = list()
clusters.append(children[children.shape[0] - 1][0])
clusters.append(children[children.shape[0] - 1][1])
# Unwind the clustering tree we build earlier until we arrive at n_target_clusters or run out of clusters
# Deciding whether to keep a cluster for analysis is based on its purity:
# purity = 100*titles_ser.value_counts()[0]/titles_ser.value_counts().sum()
# I.E. What fraction of the cluster is the title with the highest count?
# Some caveats:
# (1) Because we generally don't want to see singletons or "small" clusters, they are dropped
#     Depending on n_target_clusters and nature of the clustering, this can lead to never reaching n_target_clusters
#     Hence the if len(clusters) == 0: break
# (2) Depending on min_purity this can remove large chunks of the data and up with only very small clusters
#     There is no way obvious way around this, tuning n_target_clusters and min_purity is necessary to get good output
pure_clusters = list()
while len(pure_clusters) < n_target_clusters:
    if len(clusters) == 0:
        break
    tmp_clusters = list()
    for cluster in clusters:
        titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
        purity = 100*titles_ser.value_counts()[0]/titles_ser.value_counts().sum()
        if purity > min_purity:
            pure_clusters.append(cluster)
        else:
            left = children[cluster - children.shape[0] - 1][0]
            if left > children.shape[0] and len(clustering_tree[left]["child_indices"]) > min_clus_size:
                tmp_clusters.append(left)
            right = children[cluster - children.shape[0] - 1][1]
            if right > children.shape[0] and len(clustering_tree[right]["child_indices"]) > min_clus_size:
                tmp_clusters.append(right)
    clusters = tmp_clusters

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
    training_data_list.append(cluster_matrix[:int(train_test_frac*cluster_matrix.shape[0]), :])

training_data_matrix = np.concatenate([matrix for matrix in training_data_list], axis=0)
np.random.shuffle(training_data_matrix)
X_train, y_train = training_data_matrix[:, :-1], training_data_matrix[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print("Fitting model")
print(X_train.shape)
mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1], int((2/3)*X_train.shape[1]),
                                        len(labelenc.classes_)), max_iter=1000, verbose=True)

mlp.fit(X_train, y_train)
dump(mlp, binary_path + "MLPClassifier_model.joblib")

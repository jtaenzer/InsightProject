import pandas as pd
import numpy as np
from joblib import load, dump
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import configs.analysis_config as cfg

"""
cluster_analysis.py -- produces word cloud plots from a fitted AgglomerativeClustering model

Uses the clustering matrix (model.children_) to create dictionary holding the tree like structure of the clustering
This can then be unwound to some target number of clusters, after which word clouds are made from the singletons
in each cluster. A first, basic attempt at deciding on labels for the clusters can come from the word clouds. 
"""

# Load our fitted clustering model and the data that went into the clustering
clustering = load(cfg.binary_path + "clustering_model.joblib")
titles = load(cfg.binary_path + "titles_subsampled.joblib")
labelenc = load(cfg.binary_path + "label_encoder.joblib")
title_encoding = labelenc.classes_
data_dict = load(cfg.binary_path + "data_subsampled.joblib")
count_vectorizer = load(cfg.binary_path + "count_vectorizer.joblib")
tfidf_transformer = load(cfg.binary_path + "tfidf_transformer.joblib")
titles = titles.tolist()  # Titles was saved as a pandas series


core_skills_dict = dict()
for key in data_dict.keys():
    tmp_skills = list()
    for index, row in enumerate(data_dict[key]):
        tmp_skills.extend(row.split(", "))
    tmp_series = pd.Series(tmp_skills, dtype=str)
    key_str = title_encoding.tolist()[key]
    core_skills_dict[key_str] = tmp_series.value_counts()[:cfg.core_skills_depth].index.tolist()


children = clustering.children_
print("Clustering children shape: {}".format(children.shape))


# Generate a dictionary holding the tree-like structure of the clustering tree
# Could also be useful to have a field that holds the direct direct descendants of each cluster
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
while len(pure_clusters) < cfg.n_target_clusters:
    if len(clusters) == 0:
        break
    tmp_clusters = list()
    for cluster in clusters:
        titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
        purity = 100*titles_ser.value_counts()[0]/titles_ser.value_counts().sum()
        if purity > cfg.min_purity:
            pure_clusters.append(cluster)
        else:
            left = children[cluster - children.shape[0] - 1][0]
            if left > children.shape[0] and len(clustering_tree[left]["child_indices"]) > min_clus_size:
                tmp_clusters.append(left)
            right = children[cluster - children.shape[0] - 1][1]
            if right > children.shape[0] and len(clustering_tree[right]["child_indices"]) > min_clus_size:
                tmp_clusters.append(right)
    clusters = tmp_clusters

count = 0
unique_clusters = list()
cluster_centroid_dict = dict()
# Generate word clouds of the titles and skills from the clusters we collected above!
for cluster in pure_clusters:
    titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
    count += len(titles_ser)
    cluster_label_str = titles_ser.value_counts().index[0]
    if cluster_label_str not in unique_clusters:
        unique_clusters.append(cluster_label_str)
    cluster_label_enc = title_encoding.tolist().index(cluster_label_str)

    cluster_matrix = count_vectorizer.transform(data_dict[cluster_label_enc]).toarray()
    mask = np.sum(cluster_matrix, axis=1) > 10
    cluster_matrix = cluster_matrix[mask]
    cluster_matrix = tfidf_transformer.transform(cluster_matrix).toarray()
    centroid = np.mean(cluster_matrix, axis=0)
    cluster_centroid_dict[cluster] = {"label": cluster_label_str, "centroid": centroid}

    skills_ser = pd.Series(core_skills_dict[cluster_label_str])

    print("Cluster {}".format(cluster))
    print("Label, label count, purity: ({0}, {1}, {2})".format(cluster_label_str, titles_ser.value_counts()[0], 100*titles_ser.value_counts()[0]/len(titles_ser)))
    print("Size: {}".format(len(titles_ser)))
    print("Unique titles: {}".format(len(titles_ser.value_counts())))
    print()

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(titles_ser.value_counts().to_dict())
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(cfg.plot_path + "{0}_cluster{1}_titles_wordcloud.png".format(cluster_label_str, cluster))
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(titles_ser.value_counts()[:15])), titles_ser.value_counts()[:15].values.tolist(), align='center')
    plt.xticks(range(len(titles_ser.value_counts()[:15])), titles_ser.value_counts()[:15].index.values.tolist(), size=10, rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Titles in cluster {}".format(cluster))
    plt.ylabel("Count")
    annotate_str = "Cluster size (N profiles): {}".format(len(titles_ser))
    plt.annotate(annotate_str, xy=(0.7, 0.8), xycoords='axes fraction')
    plt.savefig(cfg.plot_path + "{0}_cluster{1}_titles_histogram.png".format(cluster_label_str, cluster))
    plt.close()
    
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(skills_ser.value_counts().to_dict())
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(cfg.plot_path + "{0}_cluster{1}_skills_wordcloud.png".format(cluster_label_str, cluster))
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(skills_ser.value_counts()[:15])), skills_ser.value_counts()[:15].values.tolist(), align='center')
    plt.xticks(range(len(skills_ser.value_counts()[:15])), skills_ser.value_counts()[:15].index.values.tolist(), size=10, rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Skills in cluster {}".format(cluster))
    plt.ylabel("Count")
    annotate_str = "Cluster size (N profiles): {}".format(len(titles_ser))
    plt.annotate(annotate_str, xy=(0.7, 0.8), xycoords='axes fraction')
    plt.savefig(cfg.plot_path + "{0}_cluster{1}_skills_histogram.png".format(cluster_label_str, cluster))
    plt.close()

dump(core_skills_dict, cfg.binary_path + "core_skills_dict_key_str.joblib")
dump(cluster_centroid_dict, cfg.binary_path + "cluster_centroid_dict.joblib")

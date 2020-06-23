import pandas as pd
import numpy as np
from joblib import load, dump
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from analysis_tools import AnalysisTools
import configs.analysis_config as cfg

"""
cluster_analysis.py -- produces word cloud and histogram plots from a fitted AgglomerativeClustering model

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

tools = AnalysisTools()

# Find the core skills for each title that we encoded before clustering
core_skills_dict = tools.build_core_skills(data_dict, title_encoding, depth=cfg.core_skills_depth)
# Re-build the tree-like structure of the clustering so we can unwind it
clustering_tree = tools.build_clustering_tree(clustering, titles, title_encoding)
# Unwind the clustering tree looking for clusters with some minimum size and purity
pure_clusters = tools.find_pure_clusters(clustering, clustering_tree,
                                         cfg.n_target_clusters, cfg.min_clus_size, cfg.min_purity)


if cfg.verbose:
    data_frac = 100*sum([len(clustering_tree[cluster]["child_titles"]) for cluster in pure_clusters])/len(titles)
    print("Found {0} clusters containing {1:.1f}% of the input data".format(len(pure_clusters), data_frac))

unique_clusters = list()
cluster_centroid_dict = dict()
# Generate word clouds of the titles and skills from the clusters we collected above!
for cluster in pure_clusters:
    titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
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
    if cfg.verbose:
        print("Cluster {}".format(cluster))
        print("Label, label count, purity: ({0}, {1}, {2})".format(cluster_label_str, titles_ser.value_counts()[0], 100*titles_ser.value_counts()[0]/len(titles_ser)))
        print("Size: {}".format(len(titles_ser)))
        print("Unique titles: {}\n".format(len(titles_ser.value_counts())))
    if cfg.plotting:
        # Make title plots
        tools.make_word_cloud(titles_ser, cluster_label_str, cluster, plot_path=cfg.plot_path)
        tools.make_histogram(titles_ser, cluster_label_str, cluster, depth=15, plot_path=cfg.plot_path)
        # Make skills plots
        tools.make_word_cloud(skills_ser, cluster_label_str, cluster, plot_path=cfg.plot_path)
        tools.make_histogram(skills_ser, cluster_label_str, cluster, depth=15, plot_path=cfg.plot_path)

dump(core_skills_dict, cfg.binary_path + "core_skills_dict_key_str.joblib")
dump(cluster_centroid_dict, cfg.binary_path + "cluster_centroid_dict.joblib")

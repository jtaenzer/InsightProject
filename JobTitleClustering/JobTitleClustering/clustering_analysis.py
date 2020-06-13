import pandas as pd
from joblib import load
from wordcloud import WordCloud
from matplotlib import pyplot as plt

"""
cluster_analysis.py -- produces word cloud plots from a fitted AgglomerativeClustering model

Uses the clustering matrix (model.children_) to create dictionary holding the tree like structure of the clustering
This can then be unwound to some target number of clusters, after which word clouds are made from the singletons
in each cluster. A first, basic attempt at deciding on labels for the clusters can come from the word clouds. 
"""

path = "./binaries/"
# How many clusters do we want to find?
n_target_clusters = 50
# Do we want to ignore small clusters?
min_clus_size = 15
# Minimum purity -- to find good quality clusters we want them to be dominated by a particular title
min_purity = 5
# Load our fitted clustering model and the data that went into the clustering
clustering = load(path + "clustering_model.joblib")
titles = load(path + "titles_clean.joblib")
data = load(path + "data_clean.joblib")
titles = titles.tolist()  # Titles was saved as a pandas series

children = clustering.children_
print("Clustering children shape: {}".format(children.shape))
# Generate a dictionary holding the tree-like structure of the clustering tree
# Could also be useful to have a field that holds the direct direct descendants of each cluster
clustering_tree = dict()
for index, row in enumerate(children):
    # Check if we've found a singleton, in that case the index is exactly whats in the clustering matrix
    # Title and skills can be taken directly from the data
    if row[0] < clustering.n_leaves_:
        titles1 = [titles[int(row[0])]]
        skills1 = data[int(row[0])].split(", ")
        indices1 = [int(row[0])]
    # If we haven't found a singleton, fill indices/titles/skills from a previous iteration of this loop!
    # Note: since we're looping through the clustering matrix in order, singletons will always be added first
    else:
        titles1 = clustering_tree[int(row[0])]["child_titles"]
        skills1 = clustering_tree[int(row[0])]["child_skills"]
        indices1 = clustering_tree[int(row[0])]["child_indices"]
    # Same as above but for the other index in the clustering matrix
    if row[1] < clustering.n_leaves_:
        titles2 = [titles[int(row[1])]]
        skills2 = data[int(row[1])].split(", ")
        indices2 = [int(row[1])]
    else:
        titles2 = clustering_tree[int(row[1])]["child_titles"]
        skills2 = clustering_tree[int(row[1])]["child_skills"]
        indices2 = clustering_tree[int(row[1])]["child_indices"]

    clustering_tree[1 + index + len(children)] = {"child_titles": titles1+titles2,
                                                  "child_indices": indices1+indices2,
                                                  "child_skills": skills1+skills2}

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

# Generate word clouds of the titles and skills from the clusters we collected above!
for cluster in pure_clusters:
    print("Plotting cluster {}".format(cluster))
    titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
    print(titles_ser.value_counts().index[0], len(titles_ser), 100*titles_ser.value_counts()[0]/titles_ser.value_counts().sum())
    print(titles_ser.value_counts().index[1], len(titles_ser), 100*titles_ser.value_counts()[1]/titles_ser.value_counts().sum())
    print(titles_ser.value_counts().index[2], len(titles_ser), 100 * titles_ser.value_counts()[2] / titles_ser.value_counts().sum())
    print()
    skills_ser = pd.Series(clustering_tree[cluster]["child_skills"], dtype=str)

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(titles_ser.value_counts().to_dict())
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path + "/plots/cluster{}_title_word_cloud.png".format(cluster))
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(titles_ser.value_counts()[:15])), titles_ser.value_counts()[:15].values.tolist(), align='center')
    plt.xticks(range(len(titles_ser.value_counts()[:15])), titles_ser.value_counts()[:15].index.values.tolist(), size=10, rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Titles in cluster {}".format(cluster))
    plt.ylabel("Count")
    annotate_str = "Cluster size (N profiles): {}".format(len(titles_ser))
    plt.annotate(annotate_str, xy=(0.7, 0.8), xycoords='axes fraction')
    plt.savefig(path + "/plots/cluster{}_titles_histogram.png".format(cluster))
    plt.close()

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(skills_ser.value_counts().to_dict())
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path + "/plots/cluster{}_skills_word_cloud.png".format(cluster))
    plt.close()

    plt.figure(figsize=(25, 10))
    plt.bar(range(len(skills_ser.value_counts()[:15])), skills_ser.value_counts()[:15].values.tolist(), align='center')
    plt.xticks(range(len(skills_ser.value_counts()[:15])), skills_ser.value_counts()[:15].index.values.tolist(), size=10, rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.title("Skills in cluster {}".format(cluster))
    plt.ylabel("Count")
    annotate_str = "Cluster size (N profiles): {}".format(len(titles_ser))
    plt.annotate(annotate_str, xy=(0.7, 0.8), xycoords='axes fraction')
    plt.savefig(path + "/plots/cluster{}_skills_histogram.png".format(cluster))
    plt.close()





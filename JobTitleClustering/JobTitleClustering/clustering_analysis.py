import pandas as pd
from joblib import load
from wordcloud import WordCloud
from matplotlib import pyplot as plt


path = "D:/FutureFit/tfidf_exploration/analysis_test_folder/"
# How many clusters do we want to find?
n_target_clusters = 4
# Load our fitted clustering model and the data that went into the clustering
clustering = load(path + "clustering_ward_euclidean.joblib")
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
# For now we are starting from the top-most cluster
clusters = list()
clusters.append(children[children.shape[0] - 1][0])
clusters.append(children[children.shape[0] - 1][1])
# Unwind the clustering tree we build earlier until we arrive at n_target_clusters
# Some caveats:
# (1) Because we generally don't want to see singletons or "small" clusters, they are dropped
#     Depending on n_target_clusters and nature of the clustering, this can lead to never reaching n_target_clusters
#     Hence the if len(clusters) == 0: break
# (2) There is no guarantee that the output for a given n_target_clusters will contain all of the data
#     Several ways to fix this -- still thinking about it
#     For now, recommend playing with n_target_clusters if data seems to be missing
while len(clusters) < n_target_clusters:
    if len(clusters) == 0:
        break
    tmp_clusters = list()
    for cluster in clusters:
        left = children[cluster - children.shape[0] - 1][0]
        right = children[cluster - children.shape[0] - 1][1]
        # Drop singletons and small clusters
        if left > children.shape[0] and len(clustering_tree[left]["child_indices"]) > 10:
            tmp_clusters.append(left)
        if right > children.shape[0] and len(clustering_tree[right]["child_indices"]) > 10:
            tmp_clusters.append(right)
        # If we exceeded n_target_clusters in this loop, remove left/right and re-append the parent cluster
        # This should lead to equality and the while loop will end
        if len(tmp_clusters) > n_target_clusters:
            tmp_clusters = tmp_clusters[:-2]
            tmp_clusters.append(cluster)
    clusters = tmp_clusters

# Generate word clouds of the titles and skills from the clusters we collected above!
for cluster in clusters:
    print("Plotting cluster {}".format(cluster))
    titles_ser = pd.Series(clustering_tree[cluster]["child_titles"], dtype=str)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(titles_ser.value_counts().to_dict())
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path + "title_word_cloud_cluster{}.png".format(cluster))
    plt.close()

    skills_ser = pd.Series(clustering_tree[cluster]["child_skills"], dtype=str)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=5).generate_from_frequencies(skills_ser.value_counts().to_dict())
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path + "skills_word_cloud_cluster{}.png".format(cluster))
    plt.close()


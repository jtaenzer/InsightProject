import os
from joblib import dump
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from pipeline import Pipeline


# Stolen from stackoverflow
def get_distances(X, model, mode='l2'):
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d

        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append( wNew)
    return distances, weights


### CONFIG
titles_to_extract = ["data scientist", "registered nurse", "marketing manager"]
min_skill_depth = 15000
min_skill_length = 10
profile_depth = 25
n_cluster_stop = 1
linkages = ['ward', 'single', 'complete', 'average']
affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
color_palette = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
save_path = "./binaries/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Create the directory to save plots and models, if it doesn't exist already
###

data_pipeline = Pipeline("FutureFitAI_database", "talent_profiles_CAN", binary_path=save_path)
print("Getting raw data from the DB")
data_pipeline.get_skills_by_titles(titles_to_extract, min_skill_length=min_skill_length, n_profiles=profile_depth)
print("Preparing data for CountVectorizer and TfidfTransformer")
data_pipeline.prepare_data_for_count_vectorizer(skill_depth=min_skill_depth)
print("Tranforming with CountVectorizer")
data_pipeline.setup_count_vectorizer_and_transform()
print("Transforming with TfidfTransformer")
data_pipeline.setup_tfidf_transformer_and_fit_transform(data_pipeline.data_count_matrix)
print("Dropping data points with too few skills")
data_pipeline.drop_matrix_rows_by_sum(min_skill_length=min_skill_length)
print("Dumping binaries")
data_pipeline.dump_binaries()

print("Clustering")
for linkage in linkages:
    for affinity in affinities:
        if linkage == "ward" and affinity != "euclidean":
            continue
        # Create and fit the model, dump output to a pickle in case we need it later
        model = AgglomerativeClustering(affinity=affinity, linkage=linkage)
        clustering = model.fit(data_pipeline.data_tfidf_matrix)
        dump(clustering, save_path + "clustering_model.joblib")

        print("Plotting {0} {1}".format(linkage, affinity))
        children = clustering.children_
        distance, weight = get_distances(data_pipeline.data_tfidf_matrix, clustering)
        linkage_matrix = np.column_stack([children, distance, weight]).astype(float)

        clustering_tree = dict()
        for index, row in enumerate(linkage_matrix):
            if row[0] < clustering.n_leaves_:
                indices_left = [int(row[0])]
            else:
                indices_left = clustering_tree[int(row[0])]["left"] + clustering_tree[int(row[0])]["right"]
            if row[1] < clustering.n_leaves_:
                indices_right = [int(row[1])]
            else:
                indices_right = clustering_tree[int(row[1])]["left"] + clustering_tree[int(row[1])]["right"]
            clustering_tree[1 + index + len(children)] = {"left": indices_left,
                                                          "right": indices_right,
                                                          "dist": row[2],
                                                          }
        coph_list = list()
        for key in sorted(clustering_tree.keys()):
            for i in clustering_tree[key]["left"]:
                for j in clustering_tree[key]["right"]:
                    vec_i = data_pipeline.data_tfidf_matrix[i]
                    vec_j = data_pipeline.data_tfidf_matrix[j]
                    dij = np.linalg.norm(vec_i - vec_j)
                    dlr = clustering_tree[key]["dist"]
                    coph_list.append([dij, dlr])
        coph_matrix = np.array(coph_list)
        print("Cophenetic correlation coefficient:", np.corrcoef(coph_matrix[:, 0], coph_matrix[:, 1]))

        plt.figure(figsize=(25, 15))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("job title")
        plt.ylabel("distance")
        dendrogram(linkage_matrix, leaf_rotation=90., leaf_font_size=5, labels=data_pipeline.titles_clean.tolist())
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            for index, title in enumerate(titles_to_extract):
                if title in lbl.get_text():
                    color = color_palette[index]
            lbl.set_color(color)
        plt.savefig(save_path + "clustering_{0}_{1}.png".format(linkage, affinity))
        plt.close()

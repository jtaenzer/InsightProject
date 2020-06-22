import numpy as np
import pandas as pd


class AnalysisTools:
    def __init__(self):
        pass


    @staticmethod
    def build_core_skills(data, title_encoding, depth=10):
        core_skills = dict()
        for key in data.keys():
            tmp_skills = list()
            for index, row in enumerate(data[key]):
                tmp_skills.extend(row.split(", "))
            tmp_series = pd.Series(tmp_skills, dtype=str)
            key_str = title_encoding.tolist()[key]
            core_skills[key_str] = tmp_series.value_counts()[:depth].index.tolist()
        return core_skills

    # Generate a dictionary holding the tree-like structure of the clustering tree
    # Could also be useful to have a field that holds the direct parent and descendants of each cluster
    @staticmethod
    def build_clustering_tree(model, titles, title_encoding):
        clustering_tree = dict()
        children = model.children_
        for index, row in enumerate(children):
            # Check if we've found a singleton, in that case the index is exactly whats in the clustering matrix
            # Title and skills can be taken directly from the data
            if row[0] < model.n_leaves_:
                titles1 = [title_encoding[int(titles[int(row[0])])]]
                indices1 = [int(row[0])]
            # If we haven't found a singleton, fill indices/titles/skills from a previous iteration of this loop!
            # Note: since we're looping through the clustering matrix in order, singletons will always be added first
            else:
                titles1 = clustering_tree[int(row[0])]["child_titles"]
                indices1 = clustering_tree[int(row[0])]["child_indices"]
            # Same as above but for the other index in the clustering matrix
            if row[1] < model.n_leaves_:
                titles2 = [title_encoding[int(titles[int(row[1])])]]
                indices2 = [int(row[1])]
            else:
                titles2 = clustering_tree[int(row[1])]["child_titles"]
                indices2 = clustering_tree[int(row[1])]["child_indices"]

            clustering_tree[1 + index + len(children)] = {"child_titles": titles1 + titles2,
                                                          "child_indices": indices1 + indices2}
        return clustering_tree

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
    @staticmethod
    def find_pure_clusters(model, clustering_tree, n_target_clusters, min_clus_size, min_purity):
        pure_clusters = list()
        children = model.children_
        # Define the starting point to unwind our clustering tree -- this could be tunable
        # For now we are starting from the top-most cluster which should contain all possible children
        clusters = list()
        clusters.append(children[children.shape[0] - 1][0])
        clusters.append(children[children.shape[0] - 1][1])
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
        return pure_clusters

    # Stolen from stackoverflow
    @staticmethod
    def get_cluster_distances(X, model, mode='l2'):
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
            if mode == 'l2':  # Increase the higher level cluster size suing an l2 norm
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

    @staticmethod
    def make_word_cloud(series_counts):
        pass

    @staticmethod
    def make_histogram(series_counts):
        pass

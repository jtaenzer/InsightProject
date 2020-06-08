from joblib import dump
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from pipeline import Pipeline

### CONFIG
titles_to_extract = ["data scientist", "registered nurse"]
min_skill_depth = 10000
min_skill_length = 5
profile_depth = 0
n_cluster_stop = 1
linkages = ['ward', 'single', 'complete', 'average']
affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
color_palette = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
save_path = "D:/FutureFit/tfidf_exploration/"
# Create the directory to save plots and models, if it doesn't exist already
###


data_pipeline = Pipeline("FutureFitAI_database", "talent_profiles", binary_path=save_path)
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
        model = AgglomerativeClustering(affinity=affinity, linkage=linkage, connectivity=connectivity)
        clustering = model.fit(data_pipeline.data_tfidf_matrix)
        dump(clustering, save_path + "clustering_{0}_{1}.joblib".format(linkage, affinity))

        print("Plotting {0} {1}".format(linkage, affinity))
        children = model.children_
        distance = np.arange(children.shape[0])
        no_of_observations = np.arange(2, children.shape[0] + 2)
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
        plt.figure(figsize=(25, 10))
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

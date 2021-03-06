import os
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from pipeline import Pipeline


# Configuration -- could go in a separate config but this is just an exploration script!
remake_data = False
do_plot = True
titles_to_extract = ["data scientist", "data engineer", "registered nurse", "marketing manager", "brand manager"]
skill_depth = 100
save_path = "D:/FutureFit/tfidf_exploration/skill_depth_{0}_{1}/".format(str(skill_depth), "_".join(titles_to_extract).replace(" ", "_"))
min_skill_length = 10
profile_depth = 100
linkages = ['ward', 'single', 'complete', 'average']
affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
color_palette = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(save_path):
    os.makedirs(save_path)

if remake_data:
    vocab_for_counts = pd.Series(dtype=str)
    titles_for_plotting = []
    data_for_counts = []
    data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
    for title in titles_to_extract:
        print("Retrieving data for title: {}".format(title))
        _, data, data_flat = data_extractor.get_skills_by_titles([title])
        # Create vocab for TF-IDF
        data_flat_ser = pd.Series(data_flat, dtype=str)
        mask = data_flat_ser.isin(data_flat_ser.value_counts()[:skill_depth].index)
        vocab_for_counts = vocab_for_counts.append(data_flat_ser[mask].drop_duplicates(), ignore_index=True)
        # Create "documents" for TF-IDF
        data_join = [" ".join(row) for row in data if len(row) > min_skill_length]
        data_for_counts.extend(data_join[:profile_depth])
        # Harmonize the titles based on what we requested from the DB
        titles_for_plotting.extend([title]*profile_depth)

    dump(data_for_counts, save_path + "data_join.joblib")
    dump(vocab_for_counts.drop_duplicates().tolist(), save_path + "vocab_for_counts.joblib")
    dump(titles_for_plotting, save_path + "titles_for_plotting.joblib")

else:
    data_for_counts = load(save_path + "data_join.joblib")
    vocab_for_counts = pd.Series(load(save_path + "vocab_for_counts.joblib"), dtype=str)
    titles_for_plotting = load(save_path + "titles_for_plotting.joblib")

print("Transforming data to count matrix")
count_vectorizer = CountVectorizer(vocabulary=vocab_for_counts.drop_duplicates().tolist())
data_count_matrix = count_vectorizer.transform(data_for_counts)
dump(count_vectorizer, save_path + "count_vectorizer.joblib")

print("Transforming data to tfidf matrix")
tfidf_transformer = TfidfTransformer()
data_tfidf_matrix = tfidf_transformer.fit_transform(data_count_matrix)
dump(tfidf_transformer, save_path + "tfidf_transformer.joblib")

# Remove empty rows so we can use the cosine distance
data_tfidf_matrix = data_tfidf_matrix.toarray()
mask = np.sum(data_tfidf_matrix, axis=1) != 0
data_tfidf_matrix = data_tfidf_matrix[mask]
titles_for_plotting = np.array(titles_for_plotting, dtype=str)
titles_for_plotting = titles_for_plotting[mask].tolist()

# For each linkage/affinity combination, cluster our  tf-idf transformed data and plot a dendrogram!
print("Clustering!")
for linkage in linkages:
    for affinity in affinities:
        if linkage == "ward" and affinity != "euclidean":
            continue
        # Create and fit the model, dump output to a pickle in case we need it later
        model = AgglomerativeClustering(affinity=affinity, linkage=linkage)
        clustering = model.fit(data_tfidf_matrix)
        dump(clustering, save_path + "clustering_{0}_{1}.joblib".format(linkage, affinity))
        # Plot the dendrogram
        if do_plot:
            children = model.children_
            distance = np.arange(children.shape[0])
            no_of_observations = np.arange(2, children.shape[0] + 2)
            linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
            plt.figure(figsize=(25, 10))
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("job title")
            plt.ylabel("distance")
            dendrogram(linkage_matrix, leaf_rotation=90., leaf_font_size=5, labels=titles_for_plotting)
            ax = plt.gca()
            xlbls = ax.get_xmajorticklabels()
            for lbl in xlbls:
                lbl.set_color(color_palette[titles_to_extract.index(lbl.get_text())])
            plt.savefig(save_path + "clustering_{0}_{1}.png".format(linkage, affinity))
            plt.close()

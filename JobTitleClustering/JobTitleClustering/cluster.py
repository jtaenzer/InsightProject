import os
from joblib import dump
from sklearn.cluster import AgglomerativeClustering
import configs.cluster_config as cfg
from pipeline import Pipeline

# Create the directory to save plots and models, if it doesn't exist already
save_path = "D:/FutureFit/svd_exploration/skill_freq_{0}_skill_length_{1}_title_freq_{2}/"\
    .format(cfg.min_skill_depth, cfg.min_skill_length, cfg.min_title_depth)
if not os.path.exists(save_path):
    os.makedirs(save_path)

data_pipeline = Pipeline("FutureFitAI_database", "talent_profiles", binary_path=save_path)
print("Getting raw data from the DB")
data_pipeline.get_all_skills_primary(min_skill_length=cfg.min_skill_length)
print("Dropping titles from bad title list from data")
data_pipeline.drop_titles_from_data(cfg.titles_to_drop, title_depth=cfg.min_title_depth)
print("Preparing data for CountVectorizer and TfidfTransformer")
data_pipeline.prepare_data_for_count_vectorizer(skill_depth=cfg.min_skill_depth)
print("Tranforming with CountVectorizer")
data_pipeline.setup_count_vectorizer_and_transform()
print("Transforming with TfidfTransformer")
data_pipeline.setup_tfidf_transformer_and_fit_transform(data_pipeline.data_count_matrix)
print("Dumping binaries")
data_pipeline.dump_binaries()
print("Dropping data points with too few skills")
data_pipeline.drop_matrix_rows_by_sum(min_skill_length=cfg.min_skill_length)
print("Pipeline complete!")


print("Clustering")
# Create and fit the model, dump output to a pickle in case we need it later
model = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=cfg.n_cluster_stop)
clustering = model.fit(data_pipeline.data_tfidf_matrix)
dump(model, save_path + "clustering_model.joblib")

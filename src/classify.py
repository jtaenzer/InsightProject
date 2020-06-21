import os
import sys
import numpy as np
from joblib import dump, load
import configs.classify_config as cfg
from pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

"""
classify.py trains a single hidden layer MLPClassifier model to predict job titles from skills

Current version extracts a "pre-labeled" dataset directly from the DB -- will need to be adapted to use cluster labels
"""

binary_path = cfg.binary_path + "skill_depth_{0}_skill_length_{1}_title_freq_{2}/".format(cfg.min_skill_depth, cfg.min_skill_length, cfg.min_title_freq)

# Create the binary directory if it doesn't exist
if not os.path.exists(cfg.binary_path):
    os.makedirs(cfg.binary_path)

data_pipeline = Pipeline("FutureFitAI_database", cfg.collection_name, binary_path=cfg.binary_path)
print("Getting raw data from the DB")
data_pipeline.get_all_skills_primary(min_skill_length=cfg.min_skill_length)
print("Dropping titles from bad title list from data")
data_pipeline.drop_titles_from_data(cfg.titles_to_drop, min_title_freq=cfg.min_title_freq)
print("Preparing data for CountVectorizer and TfidfTransformer")
data_pipeline.prepare_data_for_count_vectorizer(skill_depth=cfg.min_skill_depth)
print("Tranforming with CountVectorizer")
data_pipeline.setup_count_vectorizer_and_transform()
print("Transforming with TfidfTransformer")
data_pipeline.setup_tfidf_transformer_and_fit_transform(data_pipeline.data_count_matrix)
print("Integer encoding titles")
data_pipeline.setup_label_encoder_and_fit_transform()

# More of the code below could be moved into the pipeline
print("Splitting data by title")
data_str_dict = dict()
for title in data_pipeline.titles_encoded:
    data_str_dict[title] = []

for index, row in enumerate(data_pipeline.data_clean):
    data_str_dict[data_pipeline.titles_encoded[index]].append(row)

print("Splitting training and testing data")
data_training = dict()
data_testing = dict()
for index, key in enumerate(data_str_dict):
    if index % 10 == 0:
        sys.stdout.write("\r")
        sys.stdout.write("{:2.0f}".format(float(index / len(data_str_dict)) * 100) + "%")
    matrix = data_pipeline.tfidf_transformer.transform(data_pipeline.count_vectorizer.transform(data_str_dict[key])).toarray()
    matrix = matrix[np.sum(matrix, axis=1) != 0]
    np.random.shuffle(matrix)
    data_training[key] = matrix[:int(matrix.shape[0]*cfg.train_test_frac), :]
    titles_col = np.array([[key]*data_training[key].shape[0]]).reshape(-1, 1)
    data_training[key] = np.concatenate((data_training[key], titles_col), axis=1)

    data_testing[key] = matrix[int(matrix.shape[0]*cfg.train_test_frac):, :]
    titles_col = np.array([[key]*data_testing[key].shape[0]]).reshape(-1, 1)
    data_testing[key] = np.concatenate((data_testing[key], titles_col), axis=1)
print()


data_training_mat = np.concatenate([data_training[key] for key in data_training.keys()], axis=0)
dump(data_training_mat, cfg.binary_path + "data_training_matrix.joblib")
data_testing_mat = np.concatenate([data_testing[key] for key in data_testing.keys()], axis=0)
dump(data_testing_mat, cfg.binary_path + "data_testing_matrix.joblib")

print("Preparing model")
X_train, y_train = data_training_mat[:, :-1], data_training_mat[:, -1]
X_test, y_test = data_testing_mat[:, :-1], data_testing_mat[:, -1]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = data_pipeline.setup_standard_scaler_and_fit_transform(X_train)

print("dumping binaries")
data_pipeline.dump_binaries()

mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1], int((2/3)*X_train.shape[1]),
                                        len(data_pipeline.label_encoder.classes_)), max_iter=1000, verbose=True)
print("Fitting model")
mlp.fit(X_train, y_train)
dump(mlp, cfg.binary_path + "model.joblib")

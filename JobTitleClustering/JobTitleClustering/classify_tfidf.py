import os
import sys
import random
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from pipeline import Pipeline

remake_data = False
remake_transform = True
min_skill_freq = 10000
min_title_freq = 1000
min_skill_length = 10
titles_to_remove = ["owner", "president", "ceo", "manager", "founder", "supervisor", "business owner", "intern", "co-founder", "président", "propriétaire"]
save_path = "D:/FutureFit/classifying_tfidf_canada/skill_freq_{0}_skill_length_{1}_title_freq_{2}/"\
    .format(min_skill_freq, min_skill_length, min_title_freq)

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(save_path):
    os.makedirs(save_path)

if remake_data:
    data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
    print("Retrieving data from DB")
    titles, data, data_flat = data_extractor.get_all_skills(min_skill_length=min_skill_length)
    if len(titles) != len(data):
        print("Length of titles inconsistent with length of data, this should never happen.")
        sys.exit(2)
    # Create vocab for TF-IDF
    vocab_for_counts = pd.Series(data_flat, dtype=str)
    # Dump data to binaries so it can be re-used later
    print("Dumping data to binaries for later use")
    dump(data, save_path + "data_join.joblib")
    dump(vocab_for_counts.drop_duplicates().tolist(), save_path + "vocab_for_counts.joblib")
    dump(titles, save_path + "titles_for_plotting.joblib")
else:
    print("Loading data from binaries")
    data = load(save_path + "data_join.joblib")
    vocab_for_counts = pd.Series(load(save_path + "vocab_for_counts.joblib"), dtype=str)
    titles = load(save_path + "titles_for_plotting.joblib")

# Keep only most frequent titles
titles_ser = pd.Series(titles, dtype=str)
mask = titles_ser.isin(titles_ser.value_counts()[titles_ser.value_counts() > min_title_freq].index)
titles_reduced = titles_ser[mask]

data_clean_min_title_freq = list()
print("Cleaning data -- title min frequency")
for index, check in enumerate(mask):
    if check:
        data_clean_min_title_freq.append(data[index])

data_clean = list()
print("Cleaning data -- removing noisy titles")
for index, title in enumerate(titles_reduced):
    if title not in titles_to_remove:
        data_clean.append(data_clean_min_title_freq[index])

for drop_title in titles_to_remove:
    titles_reduced = titles_reduced[titles_reduced != drop_title]

print("Transforming data to count matrix")
mask = vocab_for_counts.isin(vocab_for_counts.value_counts()[:min_skill_freq].index)
count_vectorizer = CountVectorizer(vocabulary=vocab_for_counts[mask].drop_duplicates().tolist())
data_count_matrix = count_vectorizer.transform(data_clean)
dump(count_vectorizer, save_path + "count_vectorizer.joblib")

print("Transforming data to tfidf matrix")
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(data_count_matrix)
dump(count_vectorizer, save_path + "tfidf_transformer.joblib")

labelenc = LabelEncoder()
titles_enc = labelenc.fit_transform(titles_reduced.tolist())
dump(labelenc.classes_, save_path + "title_encoding.joblib")
data_str_dict = dict()
for title in titles_enc:
    data_str_dict[title] = []

print("Splitting data by title")
for index, row in enumerate(data_clean):
    data_str_dict[titles_enc[index]].append(row)

print("Splitting training and testing data")
data_training = dict()
data_testing = dict()
for index, key in enumerate(data_str_dict):
    if index % 10 == 0:
        sys.stdout.write("\r")
        sys.stdout.write("{:2.0f}".format(float(index / len(data_str_dict)) * 100) + "%")
    matrix = tfidf_transformer.transform(count_vectorizer.transform(data_str_dict[key])).toarray()
    matrix = matrix[np.sum(matrix, axis=1) != 0]
    np.random.shuffle(matrix)
    data_training[key] = matrix[:int(matrix.shape[0]*0.5), :]
    titles_col = np.array([[key]*data_training[key].shape[0]]).reshape(-1, 1)
    data_training[key] = np.concatenate((data_training[key], titles_col), axis=1)

    data_testing[key] = matrix[int(matrix.shape[0]*0.5):, :]
    titles_col = np.array([[key]*data_testing[key].shape[0]]).reshape(-1, 1)
    data_testing[key] = np.concatenate((data_testing[key], titles_col), axis=1)
print()

data_training_mat = np.concatenate([data_training[key] for key in data_training.keys()], axis=0)
dump(data_training_mat, save_path + "data_training.joblib")
data_testing_mat = np.concatenate([data_testing[key] for key in data_testing.keys()], axis=0)
dump(data_testing_mat, save_path + "data_testing.joblib")

print("Preparing model")
X_train, y_train = data_training_mat[:, :-1], data_training_mat[:, -1]
X_test, y_test = data_testing_mat[:, :-1], data_testing_mat[:, -1]
scaler = StandardScaler()
scaler.fit(X_train)
dump(scaler, save_path + "scaler.joblib")
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1], int((2/3)*X_train.shape[1]), len(labelenc.classes_)),
                    max_iter=1000, verbose=True)
print("Fitting model")
mlp.fit(X_train, y_train)
dump(mlp, save_path + "model.joblib")
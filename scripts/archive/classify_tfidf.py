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
from sklearn.metrics import accuracy_score
from pipeline import Pipeline

get_raw_data = True
clean_data = True
remake_transform = True
min_skill_freq = 4000
min_title_freq = 1000
min_skill_length = 5
train_test_frac = 0.1
titles_to_remove = ["owner", "president", "ceo", "manager", "founder", "supervisor", "business owner", "intern", "co-founder", "président", "propriétaire"]
save_path = "D:/FutureFit/classifying_tfidf_canada/skill_freq_{0}_skill_length_{1}_title_freq_{2}/"\
    .format(min_skill_freq, min_skill_length, min_title_freq)

# Create the directory to save plots and models, if it doesn't exist already
if not os.path.exists(save_path):
    os.makedirs(save_path)

if get_raw_data and clean_data:
    data_extractor = Pipeline("FutureFitAI_database", "talent_profiles")
    print("Retrieving data from DB")
    titles, data = data_extractor.get_all_skills(min_skill_length=min_skill_length)
    if len(titles) != len(data):
        print("Length of titles inconsistent with length of data, this should never happen.")
        sys.exit(2)
    # Dump data to binaries so it can be re-used later
    print("Dumping data to binaries for later use")
    dump(data, save_path + "data_join.joblib")
    dump(titles, save_path + "titles_for_plotting.joblib")
elif clean_data and not get_raw_data:
    print("Loading raw data from binaries")
    data = load(save_path + "data_join.joblib")
    titles = load(save_path + "titles_for_plotting.joblib")

if clean_data:
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

    print("Creating vocab for tokenization and converting data lists to strings")
    data_flat = list()
    data_clean_str = list()
    for index, row in enumerate(data_clean):
        if index % 1000 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("{:2.0f}".format(float(index / len(data_clean)) * 100) + "%")
        data_flat.extend(row)
        data_clean_str.append(" ".join(row))
    print()
    vocab_for_counts = pd.Series(data_flat, dtype=str)
    print("Dumping clean data to binaries for later use")
    dump(data_clean_str, save_path + "data_clean_str.joblib")
    dump(titles_reduced, save_path + "titles_reduced.joblib")
    dump(vocab_for_counts, save_path + "vocab_for_counts.joblib")
else:
    print("Loading cleaned data from binaries")
    titles_reduced = load(save_path + "titles_reduced.joblib")
    data_clean_str = load(save_path + "data_clean_str.joblib")
    vocab_for_counts = load(save_path + "vocab_for_counts.joblib")

if len(titles_reduced) != len(data_clean_str):
    print("Length of titles inconsistent with length of clean data, this should never happen.")
    sys.exit(2)

print("Transforming data to count matrix")
mask = vocab_for_counts.isin(vocab_for_counts.value_counts()[:min_skill_freq].index)
count_vectorizer = CountVectorizer(vocabulary=vocab_for_counts[mask].drop_duplicates().tolist())
data_count_matrix = count_vectorizer.transform(data_clean_str)
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
for index, row in enumerate(data_clean_str):
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
    data_training[key] = matrix[:int(matrix.shape[0]*train_test_frac), :]
    titles_col = np.array([[key]*data_training[key].shape[0]]).reshape(-1, 1)
    data_training[key] = np.concatenate((data_training[key], titles_col), axis=1)

    data_testing[key] = matrix[int(matrix.shape[0]*train_test_frac):, :]
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
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)
dump(scaler, save_path + "scaler.joblib")
X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(X_train.shape[1], int((2/3)*X_train.shape[1]), len(labelenc.classes_)),
                    max_iter=1000, verbose=True)
print("Fitting model")
mlp.fit(X_train, y_train)
dump(mlp, save_path + "model.joblib")
predictions = mlp.predict(X_test)
print("accuracy score:", accuracy_score(y_test, predictions))


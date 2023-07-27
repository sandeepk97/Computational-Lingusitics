from HW2 import extract_sentence_embedding
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from time import time


TRAINING_CORPUS = 'training_sentences.txt'
TEST_CORPUS = 'test_sentences.txt'

def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    labels = data['structure'].tolist()
    return sentences, labels


def remove_nans(X_train, y_train):
    imputed = []
    for X in X_train:
        imputed.append(np.nan_to_num(X))
    imputed_array = np.array(imputed)
    imputed_array[imputed_array==None] = 0
    no_nans_X = [
        imputed_array[i]
        for i, x in enumerate(y_train)
        if x!=None
    ]
    no_nans_y = [x for x in y_train if x!=None]
    return no_nans_X, no_nans_y


def train_model(X_train, y_train):
    clf = MLPClassifier(
        alpha=.05, random_state=1, max_iter=300,
        hidden_layer_sizes=(10, 10,),
        batch_size=1
        ).fit(X_train, y_train)
    return clf


print("Loading corpus")
train_sentences, train_ys = load_corpus(TRAINING_CORPUS)
# get features
print("Extracting features")
speed_features_start = time()
train_Xs = np.vstack([extract_sentence_embedding(x) for x in train_sentences])
speed_features = time() - speed_features_start
# train model
print("Training model")
train_Xs, train_ys = remove_nans(train_Xs, train_ys)
trained_model = train_model(train_Xs, train_ys)
train_predictions = trained_model.predict(train_Xs)
training_f1 = f1_score(train_predictions, train_ys, average='macro')
training_accuracy = accuracy_score(train_ys, train_predictions)
print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")
print(f"Your accuracy for predicting sentence order on the training data is {training_accuracy}")

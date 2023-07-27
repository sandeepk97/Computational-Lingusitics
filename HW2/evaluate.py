import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from collections import Counter
import HW2

TRAINING_CORPUS = 'training_sentences.txt'

def main():
    print("Loading corpus")
    train_sentences, train_ys = load_corpus(TRAINING_CORPUS)
    # get features
    print("Extracting features")
    speed_features_start = time.time()
    train_Xs, _, _ = extract_features_all_sentences(train_sentences)
    speed_features = time.time() - speed_features_start
    # get matcher labels
    print("Checking matcher accuracy")
    matcher_classifications = get_classifications(train_sentences)
    matcher_accuracy = [matcher_classifications[i]==train_ys[i] for i, _ in enumerate(train_ys)]
    # train model
    print("Training model")
    train_Xs, train_ys = remove_nans(train_Xs, train_ys)
    trained_model = train_model(train_Xs, train_ys)
    train_predictions = trained_model.predict(train_Xs)
    training_f1 = f1_score(train_predictions, train_ys, pos_label='PO')
    print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")

def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    labels = data['structure'].tolist()
    return sentences, labels
    sentences = []
    with open(filename, 'r') as f:
        for l in f:
            sentences.append(l.rstrip())
    return sentences


def extract_numeric_features(NP, sentence):
    int_feature = HW2.extract_feature_1(NP, sentence)
    float_feature = HW2.extract_feature_3(NP, sentence)
    return int_feature, float_feature


def extract_string_features(NP, sentence):
    string_features = HW2.extract_feature_2(NP, sentence)
    return string_features


def concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        ):
    DO_features = np.concatenate([DO_string_features, DO_numeric_features], axis=1)
    PO_features = np.concatenate([PO_string_features, PO_numeric_features], axis=1)
    all_features = np.concatenate([DO_features, PO_features], axis=1)
    return all_features


def extract_features_all_sentences(
        list_of_sentences,
        mode='train',
        DO_cv=None,
        PO_cv=None
        ):
    DO_numeric_features, PO_numeric_features = [], []
    DO_string_features, PO_string_features = [], []
    # TODO: feature extract is done
    for s in list_of_sentences:
        DO_NP = HW2.extract_direct_object(s)
        PO_NP = HW2.extract_indirect_object(s)
        if DO_NP is None:
            DO_NP = ''
        if PO_NP is None:
            PO_NP = ''
        DO_numeric_features.append(
            extract_numeric_features(DO_NP, s)
        )
        PO_numeric_features.append(
            extract_numeric_features(PO_NP, s)
        )
        DO_string_features.append(
            extract_string_features(DO_NP, s)
            )
        PO_string_features.append(
            extract_string_features(PO_NP, s)
        )
    if mode=='train':
        DO_cv, DO_string_features = vectorize_train(DO_string_features)
        PO_cv, PO_string_features = vectorize_train(PO_string_features)
    else:
        DO_string_features = vectorize_test(DO_string_features, DO_cv)
        PO_string_features = vectorize_test(PO_string_features, PO_cv)
    all_features = concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        )
    if mode=='train':
        return all_features, DO_cv, PO_cv
    else:
        return all_features


def dummy_tokenizer(doc):
    return [doc]


def remove_nans(X_train, y_train):
    imputed = []
    for X in X_train:
        imputed.append(np.nan_to_num(X))
    imputed_array = np.array(imputed)
    no_nans_X = [
        imputed_array[i]
        for i, x in enumerate(y_train)
        if x!=None
    ]
    no_nans_y = [x for x in y_train if x!=None]
    return no_nans_X, no_nans_y


def vectorize_train(list_of_strings: list):
    vectorizer = CountVectorizer(
        min_df=5, max_df=len(list_of_strings),
        tokenizer=dummy_tokenizer)
    counts = Counter(list_of_strings)
    strings_w_unks = []
    for w in counts:
        if counts[w] < 5:
            strings_w_unks += ['UNK'] * counts[w]
        else:
            strings_w_unks += [w] * counts[w]
    vectorized = np.array(vectorizer.fit_transform(
        strings_w_unks
        ).todense())
    return vectorizer, vectorized


def vectorize_test(list_of_strings, vectorizer):
    strings_w_unks = []
    strings_w_unks = []
    counts = Counter(list_of_strings)
    for w in counts:
        if counts[w] < 5:
            strings_w_unks += ['UNK'] * counts[w]
        else:
            strings_w_unks += [w] * counts[w]
    vectorized = np.array(vectorizer.transform(
        strings_w_unks
        ).todense())
    return vectorized


def get_classification(sentence):
    y = HW2.get_sentence_structure(sentence)
    return y


def get_classifications(list_of_sentences):
    ys = []
    for s in list_of_sentences:
        ys.append(get_classification(s))
    return ys


def train_model(X_train, y_train):
    clf = MLPClassifier(
        alpha=.05, random_state=1, max_iter=300,
        hidden_layer_sizes=(10, 10,),
        batch_size=1
        ).fit(X_train, y_train)
    return clf


main()
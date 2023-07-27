import evaluate
import HW2
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score
from time import time

TRAINING_CORPUS = 'training_sentences.txt'
TEST_CORPUS = 'test_sentences.txt'

def load_corpus(filename):
    data = pd.read_csv(filename, sep="\t")
    sentences = data['sentence'].tolist()
    labels = data['structure'].tolist()
    return sentences, labels

print("Loading corpus")
train_sentences, train_ys = load_corpus(TRAINING_CORPUS)
# get features
print("Extracting features")
speed_features_start = time()
train_Xs = [HW2.extract_sentence_embedding(x) for x in train_sentences]
speed_features = time() - speed_features_start
# get matcher labels
print("Checking matcher accuracy")
matcher_classifications = evaluate.get_classifications(train_sentences)
matcher_accuracy = [matcher_classifications[i]==train_ys[i] for i, _ in enumerate(train_ys)]
# train model
print("Training model")
train_Xs, train_ys = evaluate.remove_nans(train_Xs, train_ys)
trained_model = evaluate.train_model(train_Xs, train_ys)
train_predictions = trained_model.predict(train_Xs)
training_f1 = f1_score(train_predictions, train_ys, pos_label='PO')
print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")
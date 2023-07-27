from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
from collections import Counter

TRAINING_CORPUS = 'training_sentences.txt'



import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')

def get_sentence_structure(sentence):
   doc = nlp(sentence)
   for token in doc:
       if token.dep_ == 'dobj':
           for child in token.head.children:
               if child.dep_ in ['dative', 'obl']:
                   return 'DO'
       elif token.dep_ in ['dative', 'obl']:
           for child in token.head.children:
               if child.dep_ == 'dobj':
                   return 'DO'
   return 'PO' if 'to ' in sentence else None



def extract_direct_object(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == 'dobj':
            return token.text
    return None


def extract_indirect_object(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == 'iobj':
            return token.text
    return None


def extract_feature_1(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        if noun_phrase != None and noun_phrase in chunk.text:
            return len(chunk.text.split())
    return None


def extract_feature_2(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        if noun_phrase != None and noun_phrase in chunk.text:
            return chunk.root.pos_
    return None


def extract_feature_3(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        if noun_phrase != None and noun_phrase in chunk.text:
            return chunk.root.similarity(nlp(noun_phrase))
    return None


def extract_sentence_embedding(sentence):
    doc = nlp(sentence)
    return doc.vector


def alter_sentence(sentence):
    doc = nlp(sentence)
    new_sentence = ''
    for token in doc:
        if token.pos_ == 'NOUN':
            new_sentence += 'noun '
        else:
            new_sentence += token.text + ' '
    return new_sentence.strip()



def main():
    train_sentences = load_corpus(TRAINING_CORPUS)
    # training
    train_Xs, _, _ = extract_features_all_sentences(train_sentences)
    train_ys = get_classifications(train_sentences)
    trained_model = train_model(train_Xs, train_ys)
    predictions = trained_model.predict(train_Xs)
    training_f1 = f1_score(predictions, train_ys)
    print(f"Your F1 score for predicting sentence order on the training data is {training_f1}")


def load_corpus(filename):
    sentences = []
    with open(filename, 'r') as f:
        for l in f:
            sentences.append(l.rstrip())
    return sentences


def extract_numeric_features(NP, sentence):
    int_feature = extract_feature_1(NP, sentence)
    float_feature = extract_feature_3(NP, sentence)
    return int_feature, float_feature


def extract_string_features(NP, sentence):
    string_features = extract_feature_2(NP, sentence)
    return string_features


def concatenate_features(
        DO_numeric_features,
        DO_string_features,
        PO_numeric_features,
        PO_string_features
        ):
    # should just work with a simple hstack, or concatenate axis=0
    DO_features = np.concat([DO_string_features, DO_numeric_features])
    PO_features = np.concat([PO_string_features, PO_numeric_features])
    all_features = np.concat([DO_features, PO_features])
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
        DO_NP = extract_direct_object(s)
        PO_NP = extract_indirect_object(s)
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
        ).todense().squeeze())
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
    vectorized = vectorizer.transform(
        strings_w_unks
        ).todense().squeeze()
    return vectorized


def get_classification(sentence):
    y = get_sentence_structure(sentence)
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

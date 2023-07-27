# your imports go here
import numpy as np

# global variables (e.g., nlp modules, sentencetransformer models, dependencymatcher objects) go here
# adding globals here will mean that when we import e.g., extract_indirect_object it will still work ;)

def get_sentence_structure(sentence):
    sentence_structure = None
    assert sentence_structure in {'DO', 'PO', None}
    return sentence_structure


def extract_direct_object(sentence):
    extracted_direct_object = None
    assert type(extracted_direct_object) is str
    return extracted_direct_object


def extract_indirect_object(sentence):
    extracted_indirect_object = None
    assert type(extract_indirect_object) is str
    return extracted_indirect_object


def extract_feature_1(noun_phrase, sentence):
    feature_1 = None
    assert type(feature_1) is int
    return feature_1


def extract_feature_2(noun_phrase, sentence):
    feature_2 = ''
    assert type(feature_2) is str
    return feature_2


def extract_feature_3(noun_phrase, sentence):
    feature_3 = None
    assert type(feature_3) is float
    return feature_3


def extract_sentence_embedding(sentence):
    sentence_embedding = None
    assert type(sentence_embedding) is np.array
    return sentence_embedding


def alter_sentence(sentence):
    altered_sentence = sentence
    # add anything to change the string here
    return altered_sentence
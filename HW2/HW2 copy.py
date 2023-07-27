# your imports go here
import numpy as np
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity

# global variables (e.g., nlp modules, sentencetransformer models, dependencymatcher objects) go here
# adding globals here will mean that when we import e.g., extract_indirect_object it will still work ;)

nlp = spacy.load('en_core_web_sm')

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
           return token.subtree.text
   return None

def extract_indirect_object(sentence):
   doc = nlp(sentence)
   for token in doc:
       if token.dep_ in ['dative', 'obl']:
           return token.subtree.text
   return None

def extract_feature_1(noun_phrase, sentence):
  #returns the length of the noun phrase in number of words
  words = noun_phrase.split()
  return len(words)

def extract_feature_2(noun_phrase, sentence):
  #returns the part of speech tag sequence (e.g. pronoun, proper nouns, singular/plural)
  tag_sequence = []
  for word in noun_phrase.split():
    tag = nltk.pos_tag([word])[0][1]
    tag_sequence.append(tag)
  return '_'.join(tag_sequence)

def extract_feature_3(noun_phrase, sentence):
  #returns the maximum cosine similarity of the noun phrase to other noun phrases in the sentence
#   words = [word for word in noun_phrase.split() if word in sentence]
#   vecs = [model[word] for word in words]
#   cos_sim = max(cosine_similarity(vecs))
#   return cos_sim
# Get embeddings for all noun phrases in the sentence
    doc = nlp(sentence)
    noun_phrases = [chunk for chunk in doc.noun_chunks]
    phrase_embeddings = [np.mean([word.vector for word in phrase], axis=0) for phrase in noun_phrases]
    
    # Compute cosine similarities between target phrase and all other noun phrases
    target_embedding = np.mean([word.vector for word in nlp(noun_phrase)], axis=0).reshape(1, -1)
    similarities = cosine_similarity(target_embedding, phrase_embeddings)
    
    # Find maximum cosine similarity
    max_similarity = np.max(similarities)
    
    return max_similarity


def extract_sentence_embedding(sentence):
    sentence_embedding = None
    assert type(sentence_embedding) is np.array
    return sentence_embedding


def alter_sentence(sentence):
    altered_sentence = sentence
    # add anything to change the string here
    return altered_sentence



print(get_sentence_structure("The driver gave the grandpa the bracelet"))

# import spacy
# import numpy as np

# nlp = spacy.load('en_core_web_sm')

# def get_sentence_structure(sentence):
#    doc = nlp(sentence)
#    for token in doc:
#        if token.dep_ == 'dobj':
#            for child in token.head.children:
#                if child.dep_ in ['dative', 'obl']:
#                    return 'DO'
#        elif token.dep_ in ['dative', 'obl']:
#            for child in token.head.children:
#                if child.dep_ == 'dobj':
#                    return 'DO'
#    return 'PO' if 'to ' in sentence else None


# def extract_direct_object(sentence):
#     doc = nlp(sentence)
#     for token in doc:
#         if token.dep_ == 'dobj':
#             return token.text
#     return None


# def extract_indirect_object(sentence):
#     doc = nlp(sentence)
#     for token in doc:
#         if token.dep_ == 'iobj':
#             return token.text
#     return None


# def extract_feature_1(noun_phrase, sentence):
#     doc = nlp(sentence)
#     for chunk in doc.noun_chunks:
#         if noun_phrase in chunk.text:
#             return len(chunk.text.split())
#     return None


# def extract_feature_2(noun_phrase, sentence):
#     doc = nlp(sentence)
#     for chunk in doc.noun_chunks:
#         if noun_phrase in chunk.text:
#             return chunk.root.pos_
#     return None


# def extract_feature_3(noun_phrase, sentence):
#     doc = nlp(sentence)
#     for chunk in doc.noun_chunks:
#         if noun_phrase in chunk.text:
#             return chunk.root.similarity(nlp(noun_phrase))
#     return None


# def extract_sentence_embedding(sentence):
#     doc = nlp(sentence)
#     return doc.vector


# def alter_sentence(sentence):
#     doc = nlp(sentence)
#     new_sentence = ''
#     for token in doc:
#         if token.pos_ == 'NOUN':
#             new_sentence += 'noun '
#         else:
#             new_sentence += token.text + ' '
#     return new_sentence.strip()

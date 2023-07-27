import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
import spacy_stanza
from nltk.tree import Tree
from spacy.matcher import DependencyMatcher

nlp=spacy.load("en_core_web_sm")

nlp1=spacy_stanza.load_pipeline("en")
PO_matcher=DependencyMatcher(nlp1.vocab)
DO_matcher=DependencyMatcher(nlp1.vocab)

PO=[
    {
        "RIGHT_ID":"direct_obj",
        "RIGHT_ATTRS":{
            "DEP":"obj"
            }
        },
        {
            "LEFT_ID":"direct_obj",
            "LEFT_ATTRS":{
                "DEP":"obj"
        },
        "RIGHT_ID":"indirect_obj",
        "REL_OP":"$++",
        "RIGHT_ATTRS":{
            "DEP":"obl"
        }
    }
]
DO=[
    {
        "RIGHT_ID":"indirect_obj",
        "RIGHT_ATTRS":{
            "DEP":"iobj"
            }
        },
        {
            "LEFT_ID":"indirect_obj",
            "LEFT_ATTRS":{
                "DEP":"iobj"
        },
        "RIGHT_ID":"direct_obj",
        "REL_OP":"$++",
        "RIGHT_ATTRS":{
            "DEP":"obj"
        }
    }
]
PO_matcher.add("PO_dative",[PO])
DO_matcher.add("DO_dative",[DO])
def get_sentence_structure(sentence):   
    sentence_structure = nlp1(sentence)
    PO_matches=PO_matcher(sentence_structure)
    DO_matches=DO_matcher(sentence_structure)
    if len(PO_matches)>0:
        return "PO"
    if len(DO_matches)>0:
        return "DO"
    return None

def extract_direct_object(sentence):
    extracted_direct_object = nlp1(sentence)
    PO_matches=PO_matcher(extracted_direct_object)
    DO_matches=DO_matcher(extracted_direct_object)
    if len(PO_matches)>0:
        edo=""
        _,token_ids=PO_matches[0]
        if len(token_ids)>0:
            token=extracted_direct_object[token_ids[0]]
            if token.dep=="obj":
                for x in token.subtree:
                    edo+=str(x.test)+" "
            return edo.strip() 
    if len(DO_matches)>0:
        edo=""
        _,token_ids=DO_matches[0]
        if len(token_ids)>1:
            token=extracted_direct_object[token_ids[1]]
            if token.dep=="obj":
                for x in token.subtree:
                    edo+=str(x.test)+" "
            return edo.strip()
        return None

def extract_indirect_object(sentence):
    extracted_indirect_object = nlp1(sentence)
    DO_matches=DO_matcher(extracted_indirect_object)
    PO_matches=PO_matcher(extracted_indirect_object)
    if len(DO_matches)>0:
        eid=""
        _,token_ids=DO_matches[0]
        if len(token_ids)>0:
            token=extracted_indirect_object[token_ids[0]]
            for x in token.subtree:
                eid+=str(x.test)+" "
            return eid.strip()
    if len(PO_matches)>0:
        eid=""
        _,token_ids=PO_matches[0]
        if len(token_ids)>1:
            token=extracted_indirect_object[token_ids[1]]
            for x in token.subtree:
                eid+=str(x.test)+" "
            return eid.strip()
    return None

def extract_feature_1(noun_phrase, sentence):
    feature_1 = nltk.word_tokenize(noun_phrase)
    return len(feature_1)
   # assert type(feature_1) is int
    #return feature_1


def extract_feature_2(noun_phrase, sentence):
    feature_2 = ''
    assert type(feature_2) is str
    return feature_2
    #return int ('to' in noun_phrase.lower())


def extract_feature_3(noun_phrase, sentence):
    feature_3 = word_tokenize(sentence.lower())
    max=0.0
    for word in feature_3:
        if word == noun_phrase:
            continue
        if not any(c.isalpha() for c in word):
            continue
        word_sy=wordnet.synsets(word)
        if not word_sy:
            continue
        word_pos=word_sy[0].pos()
        if word_pos not in ['n','a']:
            continue
        np_sy = wordnet.synsets(noun_phrase)
        if not np_sy:
            continue
        np_pos = np_sy[0].pos()
        if np_pos not in ['n','a']:
            continue
        sim=wordnet.wup_similarity(np_sy[0],word_sy[0])
        if sim is not None and sim > max:
            max=sim
    return max         

model = 'all-distilroberta-v1'
m1 = SentenceTransformer(model)

def extract_sentence_embedding(sentence):
    sentence_embedding = m1.encode(sentence)
    #assert type(sentence_embedding) is np.array
    return sentence_embedding.astype(np.float32)


def alter_sentence(sentence):
    altered_sentence = sentence.split()
    new=[]
    for word in altered_sentence:
        synsets=wordnet.synsets(word,pos=wordnet.NOUN)
        if synsets:
            sy=[lemma.name() for synset in synsets for lemma in synset.lemma()]
            if sy:
                new_word=sy[0]
            else:
                new_word=word
        else:
            new_word=word
        new.append(new_word)
    sent=' '.join(new)
    return sent
    # add anything to change the string here
    #return altered_sentence
    
    
print(alter_sentence("The driver gave the gandpa the bracelet"))

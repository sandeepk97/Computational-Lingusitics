import spacy
from spacy.matcher import DependencyMatcher
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import spacy_stanza
nlp = spacy_stanza.load_pipeline("en")
nlp1 = spacy.load("en_core_web_sm")
PO_matcher = DependencyMatcher(nlp.vocab)
DO_matcher = DependencyMatcher(nlp.vocab)

PO = [  # looks for direct objects followed by indirect objects
    {"RIGHT_ID": "direct_obj",
    "RIGHT_ATTRS": {
        "DEP": "obj" # dobj in spacy
    }},
    {
    "LEFT_ID": "direct_obj",
    "LEFT_ATTRS": {
        "DEP": "obj" # dobj in spacy
    },
    "RIGHT_ID": "indirect_obj",
    "REL_OP": "$++",  # to the right and sibling
    "RIGHT_ATTRS": {
        "DEP": "obl", # dative in spacy
    }
}
]
PO_matcher.add("PO_dative", [PO])
DO = [  # looks for indirect objects followed by direct objects
    {"RIGHT_ID": "indirect_obj",
    "RIGHT_ATTRS": {
        "DEP": "iobj" # dobj in spacy
    }},
    {
    "LEFT_ID": "indirect_obj",
    "LEFT_ATTRS": {
        "DEP": "iobj" # dobj in spacy
    },
    "RIGHT_ID": "direct_obj",
    "REL_OP": "$++",  # to the right and sibling
    "RIGHT_ATTRS": {
        "DEP": "obj", # dative in spacy
    }
}
]
DO_matcher.add("DO_dative", [DO])


def get_sentence_structure(sentence):
    doc = nlp(sentence)
    PO_matches = PO_matcher(doc)
    DO_matches = DO_matcher(doc)
    if len(PO_matches) > 0:
        return "PO"
    
    if len(DO_matches) > 0:
        return "DO"
    
    return None

def extract_direct_object(sentence):
    doc = nlp(sentence)
    
    PO_matches = PO_matcher(doc)
    if len(PO_matches) > 0:
        direct_str = ''
        _, token_ids = PO_matches[0]
        if len(token_ids) > 0:
            token = doc[token_ids[0]]
            if token.dep_ == 'obj':
                for x in token.subtree:
                    direct_str+=str(x.text)+" "
        return direct_str.strip()

    DO_matches = DO_matcher(doc)
    if len(DO_matches) > 0:
        direct_str = ''
        _, token_ids = DO_matches[0]
        if len(token_ids) > 1:
            token = doc[token_ids[1]]
            if token.dep_ == 'obj':
                for x in token.subtree:
                    direct_str+=str(x.text)+" "
        return direct_str.strip()
    return None



def extract_indirect_object(sentence):
    doc = nlp(sentence)
        
    DO_matches = DO_matcher(doc)
    if len(DO_matches) > 0:
        direct_str = ''
        _, token_ids = DO_matches[0]
        if len(token_ids) > 0:
            token = doc[token_ids[0]]
            for x in token.subtree:
                direct_str+=str(x.text)+" "
            return direct_str.strip()
    
    PO_matches = PO_matcher(doc)
    if len(PO_matches) > 0:
        direct_str = ''
        _, token_ids = PO_matches[0]
        if len(token_ids) > 1:
            token = doc[token_ids[1]]
            for x in token.subtree:
                direct_str+=str(x.text)+" "
            return direct_str.strip()
    return None



def extract_feature_1(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        # if noun_phrase != None and noun_phrase in chunk.text:
        return len(chunk.text.split())
    return 0


def extract_feature_2(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        # if noun_phrase != None and noun_phrase in chunk.text:
        return chunk.root.pos_
    return ""

def extract_feature_3(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        # if noun_phrase != None and noun_phrase in chunk.text:
        return chunk.root.similarity(nlp(noun_phrase))
    return 0.0


def extract_sentence_embedding(sentence):
    doc = nlp1(sentence)
    return doc.vector


def alter_sentence(sentence):
    doc = nlp1(sentence)
    embedding = doc.vector
    
    synonyms = []
    for token in doc:
        if token.has_vector:
            for syn in wn.synsets(token.text):
                for lemma in syn.lemmas():
                    if lemma.name() != token.text:
                        synonyms.append(lemma.name().replace('_', ' '))

    similarities = []
    for synonym in synonyms:
        synonym_embedding = nlp1(synonym).vector
        similarity = cosine_similarity([embedding], [synonym_embedding])
        similarities.append(similarity[0][0])
    
    max_similarity_index = similarities.index(max(similarities))
    new_word = synonyms[max_similarity_index]
    
    new_sentence = sentence.replace(doc[max_similarity_index].text, new_word)
    
    return new_sentence

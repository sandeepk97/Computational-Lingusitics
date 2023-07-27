# import spacy
from spacy.matcher import DependencyMatcher

# nlp = spacy.load('en_core_web_sm')
import spacy_stanza
# stanza.download("en")
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy_stanza.load_pipeline("en")

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



# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

# def extract_direct_object(sentence):
#     doc = nlp(sentence)

#     for token in doc:
#         if token.dep_ == 'dobj':
#             return token.text

#         # check for double object construction with a dative object
#         if token.dep_ == 'dative' and token.head.dep_ == 'verb':
#             for child in token.head.children:
#                 if child.dep_ == 'dobj':
#                     return child.text

#     return None

# nlp1 =spacy.load("en_core_web_sm")
def extract_direct_object(sentence):
    # doc = nlp1(sentence)
    # for token in doc:
    #     if token.dep == "dobj":
    #         return token.text
    # return None
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
            # if token.dep_ == 'obl':
            for x in token.subtree:
                direct_str+=str(x.text)+" "
            return direct_str.strip()
    
    PO_matches = PO_matcher(doc)
    if len(PO_matches) > 0:
        direct_str = ''
        _, token_ids = PO_matches[0]
        if len(token_ids) > 1:
            token = doc[token_ids[1]]
            # if token.dep_ == 'obl':
            for x in token.subtree:
                direct_str+=str(x.text)+" "
            return direct_str.strip()
    
    return None

    # doc = nlp1(sentence)
    # for token in doc:
    #     if token.dep in ["dative", "iobj", "pobj", "oprd"]:
    #         return token.text
    # return None
    
    # PO_matches = PO_matcher(doc)
    # if len(PO_matches) > 0:
    #     direct_str = ''
    #     _, token_ids = PO_matches[0]
    #     for i in range(len(token_ids)):
    #         token = doc[token_ids[i]]
    #         if token.dep_ == 'iobj':
    #             for x in token.subtree:
    #                 direct_str+=str(x.text)+" "
    #     return direct_str.strip()
    # return None
    # doc = nlp(sentence)
    # for token in doc:
    #     if token.dep_ == 'iobj':
    #         return token.text
    # return None
    # from spacy.matcher import Matcher
    # doc = nlp(sentence)
    # matcher = Matcher(nlp.vocab)
    # pattern = [{"DEP": "iobj"}]
    # matcher.add("DirectObject", [pattern])
    # matches = matcher(doc)
    # if matches:
    #     for match_id, start, end in matches:
    #         return doc[start:end].text
    # return None



def extract_feature_1(noun_phrase, sentence):
    # doc = nlp(sentence)
    # for chunk in doc.noun_chunks:
    #     # if noun_phrase != None and noun_phrase in chunk.text:
    #     return len(chunk.text.split())
    # return 0
    tokens = nltk.word_tokenize(sentence)
    np_tokens = nltk.word_tokenize(noun_phrase)
    np_tags = nltk.pos_tag(np_tokens)
    
    # Length of the noun phrase in number of words
    np_word_count = len(np_tokens)
    # return np_word_count
    
    # Length of the noun phrase in number of characters
    np_char_count = len(noun_phrase)
    # return np_char_count
    
    # Syntactic depth of noun phrase
    depth = 0
    for (word, tag) in np_tags:
        if tag.startswith('N'):
            depth += 1
    # return depth
    
    # Whether the word "to" is present (1) or not (0)
    to_present = int("to" in np_tokens)
    return to_present
    
    # Whether the head of a noun phrase is animate (1) or not (0)
    head_noun = ''
    for (word, tag) in np_tags[::-1]:
        if tag.startswith('NN'):
            head_noun = word
            break
    head_is_animate = int(1 if (wn.synsets(head_noun, pos=wn.NOUN) and wn.synset(wn.synsets(head_noun, pos=wn.NOUN)[0].name()).root_hypernyms()[0].name() in ['person.n.01', 'animal.n.01']) else 0)
    # return head_is_animate
    # Wordnet-based semantic features
    is_vertebrate = int(1 if wn.synsets(head_noun, pos=wn.NOUN) and wn.synset(wn.synsets(head_noun, pos=wn.NOUN)[0].name()).lowest_common_hypernyms(wn.synset('vertebrate.n.01'))[0].name() == 'vertebrate.n.01'  else 0)
    # return is_vertebrate
    return np_word_count#, np_char_count, depth, to_present, head_is_animate, is_vertebrate


def extract_feature_2(noun_phrase, sentence):
    # doc = nlp(sentence)
    # for chunk in doc.noun_chunks:
    #     # if noun_phrase != None and noun_phrase in chunk.text:
    #     return chunk.root.pos_
    # return ""
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)
    np_tokens = nltk.word_tokenize(noun_phrase)
    np_tags = nltk.pos_tag(np_tokens)
    
    # Part-of-speech tag sequence
    pos_tags = [tag for (word, tag) in np_tags]
    pos_tag_seq = ' '.join(pos_tags)
    # return pos_tag_seq
    
    # Verb string or lemma
    verb = ''
    for (word, tag) in tagged_tokens:
        if tag.startswith('VB'):
            verb = word
            break
    # return verb
            
    # Head noun string or lemma
    head_noun = ''
    for (word, tag) in np_tags[::-1]:
        if tag.startswith('NN'):
            head_noun = word
            break
    # return head_noun
    
    # The head noun part-of-speech
    head_noun_pos = ''
    for (word, tag) in np_tags[::-1]:
        if tag.startswith('NN'):
            head_noun_pos = tag
            break
    return head_noun_pos
    
    return pos_tag_seq, verb, head_noun, head_noun_pos



def extract_feature_3(noun_phrase, sentence):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        return chunk.root.similarity(nlp(noun_phrase))
    
    tokens = nltk.word_tokenize(sentence)
    unigrams = list(nltk.ngrams(tokens, 1))
    bigrams = list(nltk.ngrams(tokens, 2))
    trigrams = list(nltk.ngrams(tokens, 3))

    unigram_model = nltk.probability.LaplaceProbDist(nltk.FreqDist(unigrams))
    bigram_model = nltk.probability.LaplaceProbDist(nltk.FreqDist(bigrams))
    trigram_model = nltk.probability.LaplaceProbDist(nltk.FreqDist(trigrams))

    np_tokens = nltk.word_tokenize(noun_phrase)
    np_unigrams = list(nltk.ngrams(np_tokens, 1))
    np_bigrams = list(nltk.ngrams(np_tokens, 2))
    np_trigrams = list(nltk.ngrams(np_tokens, 3))

    np_unigram_prob = sum(unigram_model.prob(ug) for ug in np_unigrams) / len(np_unigrams) if len(np_unigrams) != 0 else 0
    np_bigram_prob = sum(bigram_model.prob(bg) for bg in np_bigrams) / len(np_bigrams) if len(np_bigrams) != 0 else 0
    np_trigram_prob = sum(trigram_model.prob(tg) for tg in np_trigrams) / len(np_trigrams) if len(np_trigrams) != 0 else 0
    return (np_unigram_prob + np_bigram_prob + np_trigram_prob) / 3




import spacy
nlp_a = spacy.load("en_core_web_sm")

def extract_sentence_embedding(sentence):
    doc = nlp_a(sentence)
    return doc.vector


# from nltk.corpus import wordnet as wn
# from scipy.spatial.distance import cosine

# def alter_sentence(sentence):
#     nlp = spacy_stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', use_gpu=False)
#     doc = nlp(sentence)
#     embedding = doc.vector
    
#     synonyms = []
#     for token in doc:
#         if token.has_vector:
#             for syn in wn.synsets(token.text):
#                 for lemma in syn.lemmas():
#                     if lemma.name() != token.text:
#                         synonyms.append(lemma.name().replace('_', ' '))

#     similarities = []
#     for synonym in synonyms:
#         synonym_embedding = nlp(synonym).vector
#         similarity = 1 - cosine(embedding, synonym_embedding)
#         similarities.append(similarity)
    
#     max_similarity_index = similarities.index(max(similarities))
#     new_word = synonyms[max_similarity_index]
    
#     new_sentence = sentence.replace(doc[max_similarity_index].text, new_word)
#     return new_sentence




from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nlp = spacy.load("en_core_web_sm")
# import embeddingHelper


def alter_sentence(sentence):
    # return embeddingHelper.alter_sentence1(sentence=sentence)
    # nlp1 = spacy.load("en_core_web_sm")
    doc = nlp_a(sentence)
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
        synonym_embedding = nlp_a(synonym).vector
        similarity = cosine_similarity([embedding], [synonym_embedding])
        similarities.append(similarity[0][0])
    
    max_similarity_index = similarities.index(max(similarities))
    new_word = synonyms[max_similarity_index]
    
    new_sentence = sentence.replace(doc[max_similarity_index].text, new_word)
    
    return new_sentence




# def alter_sentence(sentence):
#     doc = nlp(sentence)
#     new_sentence = ''
#     for token in doc:
#         if token.pos_ == 'NOUN':
#             new_sentence += 'noun '
#         else:
#             new_sentence += token.text + ' '
#     return new_sentence.strip()


print(get_sentence_structure("The driver gave the grandpa the bracelet"))
print(get_sentence_structure("The driver gave the bracelet to the grandpa"))
print(extract_direct_object("They gave new medication to the patient."))
# print(extract_indirect_object("They gave new medication to the patient."))
print(extract_indirect_object("What you get is that we don't turn you in to the police	you	to the police"))
# print(extract_indirect_object("We're moving him to the hospital in	him	to the hospital"))

print(alter_sentence("The driver gave the gandpa the bracelet"))


# 0.8823529411764706 f1 np_word_count f2 pos_tag_seq f3 

# 0.8823529411764706

# 0.954248366013072  np_word_count verb
# 0.954248366013072 

# 0.84 char_count verb

# 0.73 depth verb

# 0.9668874172185431 to_present verb

# 0.9864864864864865 to_present head_noun

# 0.9931972789115647 to_present head_noun_pos



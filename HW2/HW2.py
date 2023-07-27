import spacy
from spacy.matcher import DependencyMatcher
import spacy_stanza
# stanza.download("en")
import nltk
import torch
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
    PO_matched = len(PO_matches) > 0
    if PO_matched:
        return "PO"
    DO_matches = DO_matcher(doc)
    DO_matched = len(DO_matches) > 0
    if DO_matched:
        return "DO"
    return None

def extract_direct_object(sentence):
    doc = nlp(sentence)
    PO_matches = PO_matcher(doc)
    PO_matched = len(PO_matches) > 0
    if PO_matched:
        result = ""
        token_ids = PO_matches[0][1]
        if len(token_ids) > 0:
            token = doc[token_ids[0]]
            if token.dep_ == 'obj':
                for x in token.subtree:
                    result+=str(x.text)+" "
        return result.strip()

    DO_matches = DO_matcher(doc)
    DO_matched = len(DO_matches) > 0
    if DO_matched:
        result = ""
        token_ids = DO_matches[0][1]
        if len(token_ids) > 1:
            token = doc[token_ids[1]]
            if token.dep_ == 'obj':
                for x in token.subtree:
                    result+=str(x.text)+" "
        return result.strip()
    return None


def extract_indirect_object(sentence):
    doc = nlp(sentence)
        
    DO_matches = DO_matcher(doc)
    DO_matched = len(DO_matches) > 0
    if DO_matched:
        result = ""
        token_ids = DO_matches[0][1]
        if len(token_ids) > 0:
            token = doc[token_ids[0]]
            for x in token.subtree:
                result+=str(x.text)+" "
            return result.strip()
    
    PO_matches = PO_matcher(doc)
    PO_matched = len(PO_matches) > 0
    if PO_matched:
        result = ""
        token_ids = PO_matches[0][1]
        if len(token_ids) > 1:
            token = doc[token_ids[1]]
            for x in token.subtree:
                result+=str(x.text)+" "
            return result.strip()
    
    return None


def extract_feature_1(noun_phrase, sentence):
    np_tokens = nltk.word_tokenize(noun_phrase)
    # np_tags = nltk.pos_tag(np_tokens)
    
    # # Length of the noun phrase in number of words
    # np_word_count = len(np_tokens)
    # # return np_word_count
    
    # # Length of the noun phrase in number of characters
    # np_char_count = len(noun_phrase)
    # # return np_char_count
    
    # # Syntactic depth of noun phrase
    # depth = 0
    # for (word, tag) in np_tags:
    #     if tag.startswith('N'):
    #         depth += 1
    # # return depth
    
    # # Whether the word "to" is present (1) or not (0)
    to_present = int("to" in np_tokens)
    return to_present
    
    # Whether the head of a noun phrase is animate (1) or not (0)
    head_noun = ""
    for (word, tag) in np_tags[::-1]:
        if tag.startswith('NN'):
            head_noun = word
            break
    head_is_animate = int(1 if (wn.synsets(head_noun, pos=wn.NOUN) and wn.synset(wn.synsets(head_noun, pos=wn.NOUN)[0].name()).root_hypernyms()[0].name() in ['person.n.01', 'animal.n.01']) else 0)
    is_vertebrate = int(1 if wn.synsets(head_noun, pos=wn.NOUN) and wn.synset(wn.synsets(head_noun, pos=wn.NOUN)[0].name()).lowest_common_hypernyms(wn.synset('vertebrate.n.01'))[0].name() == 'vertebrate.n.01'  else 0)


def extract_feature_2(noun_phrase, sentence):
    # tokens = nltk.word_tokenize(sentence)
    # tagged_tokens = nltk.pos_tag(tokens)
    np_tokens = nltk.word_tokenize(noun_phrase)
    np_tags = nltk.pos_tag(np_tokens)
    
    # # Part-of-speech tag sequence
    # pos_tags = [tag for (word, tag) in np_tags]
    # pos_tag_seq = ' '.join(pos_tags)
    # # return pos_tag_seq
    
    # # Verb string or lemma
    # verb = ""
    # for (word, tag) in tagged_tokens:
    #     if tag.startswith('VB'):
    #         verb = word
    #         break
    # # return verb
            
    # # Head noun string or lemma
    # head_noun = ""
    # for (word, tag) in np_tags[::-1]:
    #     if tag.startswith('NN'):
    #         head_noun = word
    #         break
    # # return head_noun
    
    # The head noun part-of-speech
    head_noun_pos = ""
    for (_, tag) in np_tags[::-1]:
        if tag.startswith('NN'):
            head_noun_pos = tag
            break
    return head_noun_pos
    


def extract_feature_3(noun_phrase, sentence):
    doc = nlp(sentence)
    sim = -1
    for chunk in doc.noun_chunks:
        sim = max(sim, chunk.root.similarity(nlp(noun_phrase)))
    
    if (sim != -1): return sim
    
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
nlp_spacy = spacy.load("en_core_web_sm")

# def extract_sentence_embedding(sentence):
#     doc = nlp_spacy(sentence)
#     return doc.vector

# from sentence_transformers import SentenceTransformer

# def extract_sentence_embedding(sentence):
#     sentences = [sentence]
#     model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
#     return model.encode(sentences)

from transformers import DistilBertModel, DistilBertTokenizer
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def extract_sentence_embedding(sentence):
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        last_hidden_states = distilbert(input_ids)[0]
    sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()

    return sentence_embedding

from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nlp = spacy.load("en_core_web_sm")
# import embeddingHelper


# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

# def alter_sentence(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt")
#     outputs = model(**inputs)
#     predictions = outputs.logits.argmax(dim=-1)
#     return tokenizer.decode(predictions[0])


def alter_sentence(sentence):
    altered_sentence = sentence.split()
    new_words = []
    for word in altered_sentence:
        synsets = wn.synsets(word, pos=wn.NOUN)
        if len(synsets) > 0:
            synonyms = []
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != word:
                        synonym = lemma.name().replace('_', ' ')
                        synonyms.append(synonym)
            if len(synonyms) == 0:
                new_word = word
            else:
                new_word = synonyms[0]
        else:
            new_word = word
        new_words.append(new_word)
    return ' '.join(new_words)


print(get_sentence_structure("The driver gave the grandpa the bracelet"))
print(get_sentence_structure("The driver gave the bracelet to the grandpa"))
print(extract_direct_object("They gave new medication to the patient."))
# print(extract_indirect_object("They gave new medication to the patient."))
print(extract_indirect_object("What you get is that we don't turn you in to the police	you	to the police"))
# print(extract_indirect_object("We're moving him to the hospital in	him	to the hospital"))
print(alter_sentence("What you get is that we don't turn you in to the police	you	to the police"))

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



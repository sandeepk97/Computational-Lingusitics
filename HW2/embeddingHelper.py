import spacy
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
nlp1 = spacy.load("en_core_web_sm")

def alter_sentence1(sentence):
    # nlp = spacy.load("en_core_web_sm")
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

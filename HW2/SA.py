import spacy
import stanza
from spacy import displacy

# Load Spacy parser
nlp_spacy = spacy.load("en_core_web_sm")
doc_spacy = nlp_spacy("The doctor gave the spotted lemon to the doctor")
displacy.render(doc_spacy, style="dep", jupyter=True)

# Load Spacy-Stanza parser
nlp_stanza = stanza.Pipeline(lang="en")
doc_stanza = nlp_stanza("The doctor gave the spotted lemon to the doctor")
displacy.render(doc_stanza, style="dep", jupyter=True)

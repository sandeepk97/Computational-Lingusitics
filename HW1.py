from typing import List
from nltk import word_tokenize # the nltk word tokenizer
from spacy.lang.en import English
import torch
# import torch # for the spacy tokenizer
nlp = English() 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from torch.nn import Softmax
sm = Softmax(dim=2)
import json, math
from collections import Counter, defaultdict

# spacy things


# gpt2 tokenizer

# gpt2 model

# softmax


def load_corpus(filename: str):
    corpus = []
    with open(filename) as file:
        for line in file:
            corpus.append(line.rstrip())
    return corpus


def nltk_tokenize(sentence: str):
    return word_tokenize(sentence) # must be type list of strings


def spacy_tokenize(sentence: str):
    return [w.orth_ for w in nlp(sentence)] # must be type list of strings

def tokenize(sentence: str):
    # wrapper around whichever tokenizer you liked better
    wrapped_output = word_tokenize(sentence)
    return wrapped_output

# part of solution to 2a
def count_bigrams(corpus: list):
    bigrams_frequencies = defaultdict(int)
    for line in corpus:
        tokenized_line = tokenize(line)
        for i, word in enumerate(tokenized_line):
            if i < (len (tokenized_line) - 1):
                next_word = tokenized_line[i + 1]
            else:
                next_word = 'EOS'
            bigram = (word , next_word)
            bigrams_frequencies[bigram] += 1
    return bigrams_frequencies


# part of solution to 2a
def count_trigrams(corpus: list):
    trigrams_frequencies = defaultdict(int)
    for line in corpus:
        tokenized_line = tokenize(line)
        for i, word in enumerate(tokenized_line):
            if i < (len(tokenized_line) - 2):
                next_word, next_next_word = tokenized_line[i + 1],  tokenized_line[i + 2]
            elif i == len(tokenized_line) -2:
                next_word, next_next_word = tokenized_line[i + 1], 'EOS'
            else:
                next_word, next_next_word = 'EOS', 'EOS'
            trigram = (word,  next_word, next_next_word)
            trigrams_frequencies[trigram] += 1
    return trigrams_frequencies


# part of solution to 2b
def bigram_frequency(bigram: str, bigram_frequency_dict: dict):
    frequency_of_bigram = 0
    key = tuple(bigram.split())
    if key in bigram_frequency_dict:
        frequency_of_bigram = bigram_frequency_dict[key]
    return frequency_of_bigram


# part of solution to 2c
def trigram_frequency(trigram: str, trigram_frequency_dict: dict):
    frequency_of_trigram = 0
    key = tuple(trigram.split())
    if key in trigram_frequency_dict:
        frequency_of_trigram = trigram_frequency_dict[key]
    return frequency_of_trigram
    

# part of solution to 2d
def get_total_frequency(ngram_frequencies: dict):
    # compute the frequency of all ngrams from dictionary of counts
    total_frequency = sum(ngram_frequencies.values())
    return total_frequency


# part of solution to 2e
def get_probability(
        ngram: str,
        ngram_frequencies: dict):
    probability = 0.0
    key = tuple(ngram.split())
    if key in ngram_frequencies:
        probability = (float) (ngram_frequencies[key])/get_total_frequency(ngram_frequencies=ngram_frequencies)
    return probability


# part of solution to 3a
def forward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    fw_prob = (float) (trigram_counts[tuple(seq_of_three_tokens)])/ bigram_counts[tuple(seq_of_three_tokens[:2])]
    return fw_prob


# part of solution to 3b
def backward_transition_probability(
        seq_of_three_tokens: list,
        trigram_counts: dict,
        bigram_counts: dict
        ):
    bw_prob = (float) (trigram_counts[tuple(seq_of_three_tokens)])/ bigram_counts[tuple(seq_of_three_tokens[1:])]
    return bw_prob


# part of solution to 3c
def compare_fw_bw_probability(fw_prob: float, bw_prob: float):
    equivalence_test = fw_prob == bw_prob
    return equivalence_test


# part of solution to 3d
def sentence_likelihood(
    sentence,  # an arbitrary string
    bigram_counts,   # the output of count_bigrams
    trigram_counts   # the output of count_trigrams
    ):
    likelihood = 0.0
    
    tokens = sentence.split()
    
    for i in range(len(tokens)-2):
        trigram_count = trigram_counts[(tokens[i], tokens[i+1], tokens[i+2])] if (tokens[i], tokens[i+1], tokens[i+2]) in trigram_counts else 0
        
        bigram_count =  bigram_counts[(tokens[i], tokens[i+1])] if (tokens[i], tokens[i+1]) in bigram_counts else 0
        
        probability = trigram_count / bigram_count if bigram_count > 0 else 0.0
            
        if probability > 0:
            likelihood += math.log(probability)
        else:
            likelihood += float('-inf')
            
    return likelihood


# 4a
def neural_tokenize(sentence: str):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer_output = tokenizer(sentence, return_tensors='pt')
    return tokenizer_output



# 4b
def neural_logits(tokenizer_output):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    logits = model(tokenizer_output['input_ids'], attention_mask=tokenizer_output['attention_mask'])[0]

    return logits


# 4c
def normalize_probability(logits):
    softmax_logits = F.softmax(logits, dim=-1)
    return softmax_logits


# 4d.
def neural_fw_probability(
    softmax_logits,
    tokenizer_output
    ):
    probabilities = []

    for i, input in enumerate(softmax_logits[:, :, tokenizer_output['input_ids']][0]):
        probabilities.append(input[0][i])
    
    return torch.Tensor(probabilities)


# 4d.ii
def neural_likelihood(diagonal_of_probs):
    likelihood = None
    probs_tensor = torch.Tensor(diagonal_of_probs)
    likelihood = probs_tensor.log().sum()
    return likelihood



# line = ['Radish is a cool cat.']
# print(count_trigrams(line))
# # print( get_total_frequency(count_bigrams(line)))
# # print( get_total_frequency(count_trigrams(line)))
# tokenize_out = neural_tokenize(line)
# softmax_logits = normalize_probability(neural_logits(tokenize_out))
# print(neural_fw_probability(softmax_logits, tokenize_out))
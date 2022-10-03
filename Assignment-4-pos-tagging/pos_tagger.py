from collections import defaultdict
from pprint import pprint
import nltk
import numpy as np
import math
from viterbi import viterbi


tagged_sentence_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
"""
Create the transmission frequencies
Dict is initialized to 1 for smoothing
"""
transmission_frequency = defaultdict(lambda: defaultdict(lambda:1))
for sentence in tagged_sentence_corpus:
    for index in range(len(sentence)):
        if index == 0:
            prev_pos = 'START'
        else:
            prev_pos = sentence[index-1][1]

        current_pos = sentence[index][1]

        transmission_frequency[prev_pos][current_pos] += 1     

"""
Create the emission frequencies dictionary
Use only words that present in the test sentence
Dict is initialized to 1 for smoothing
"""
test_sentence = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
test_words = set(word[0].lower() for sentence in test_sentence for word in sentence)

emission_frequency = defaultdict(lambda: defaultdict(lambda:1))
for sentence in tagged_sentence_corpus:
    for pos in sentence:
        word = pos[0].lower()
        word_pos = pos[1]
        if word == 'preparatory':
            print(word, word_pos)
        if word not in test_words:
            continue
        else:
            emission_frequency[word][word_pos] += 1


"""
Wrapper function to run POS tagger using Viterbi algorithm
"""
def run_pos_tagger(test_sentences):
    for sentence in test_sentences:
        word_list = [word[0].lower() for word in sentence]
        pos_list = [word[1] for word in sentence]
        output = viterbi(word_list, transmission_frequency, emission_frequency)
        print("Input: ")
        print(sentence)
        print("Result: ")
        print(output)
        print('#'*50)

# print('$'*50)
# print('preparatory', emission_frequency['preparatory']) # Unknown word
# print('coming', emission_frequency['coming']) # Coming has higher Verb count, but DET to Noun is 15k, DET to very is 1.5k
# print('to', emission_frequency['to']) 
# print('introductory', emission_frequency['introductory'])
# print('face-to-face', emission_frequency['face-to-face']) # Unknown word . -> 15.7k
# print('one', emission_frequency['one']) 
# print('DET', transmission_frequency['DET'])
# print('NOUN', transmission_frequency['NOUN'])
# print('ADJ', transmission_frequency['ADJ'])
# print('$'*50)
if __name__ == '__main__':
    """
    Unknown words - preparatory
    """
    test_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    run_pos_tagger(test_sentences)



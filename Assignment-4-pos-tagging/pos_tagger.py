from collections import defaultdict
import nltk
from viterbi import viterbi

"""
Create the transition frequencies
Dict is initialized to 1 for smoothing
"""
def create_transition_frequency_dict(tagged_sentence_corpus):
    transition_frequency = defaultdict(lambda: defaultdict(lambda:1))
    for sentence in tagged_sentence_corpus:
        for index in range(len(sentence)):
            if index == 0:
                prev_pos = 'START'
            else:
                prev_pos = sentence[index-1][1]

            current_pos = sentence[index][1]

            transition_frequency[prev_pos][current_pos] += 1     
    return transition_frequency

"""
Create the emission frequencies dictionary
Use only words that present in the test sentence
Dict is initialized to 1 for smoothing
"""
def create_emission_frequency_dict(tagged_sentence_corpus):
    emission_frequency = defaultdict(lambda: defaultdict(lambda:1))
    for sentence in tagged_sentence_corpus:
        for pos in sentence:
            word = pos[0].lower()
            word_pos = pos[1]
            emission_frequency[word][word_pos] += 1
    
    return emission_frequency

"""
Wrapper function to run part-of-speech tagger using Viterbi algorithm
"""
def run_pos_tagger(tagged_sentence_corpus, test_sentences):
    transition_frequency = create_transition_frequency_dict(tagged_sentence_corpus)
    emission_frequency = create_emission_frequency_dict(tagged_sentence_corpus)

    for sentence in test_sentences:
        word_list = [word[0].lower() for word in sentence]
        pos_list = [word[1] for word in sentence]
        output = viterbi(word_list, transition_frequency, emission_frequency)
        print ("{:<20} {:<15} {:<15}".format("input_word", "actual_pos", "predicted_pos"))
        for input_word, input_pos, predicted_pos in zip(word_list, pos_list, output):
            print ("{:<20} {:<15} {:<15}".format(input_word, input_pos, predicted_pos))
        print('#'*30)


if __name__ == '__main__':
    tagged_sentence_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    test_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    run_pos_tagger(tagged_sentence_corpus, test_sentences)



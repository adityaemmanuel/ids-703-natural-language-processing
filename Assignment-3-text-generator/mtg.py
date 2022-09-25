
import numpy as np
import nltk
def finish_sentence(sentence, n, corpus, deterministic=False):
    word_list = list(nltk.ngrams(corpus, n))
    print(type(word_list))
    for word in word_list:
        print(word)
        break
    vocab_size = len(word_list)
    print(vocab_size)
    n_gram_table = np.zeros((vocab_size, vocab_size))





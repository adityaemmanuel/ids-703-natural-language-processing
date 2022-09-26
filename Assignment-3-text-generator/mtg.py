"""
Natural Language Processing - Assignment 2
Text Generation using N-gram Markov Text Generators
"""
from collections import defaultdict, Counter
import string
import random
import sys

def most_frequent(List):
    """
    Function to get the most common 
    """
    
    punctuation_list = list(string.punctuation)
    List = [x for x in List if x not in punctuation_list]
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def finish_sentence(sentence, n, corpus, deterministic=False):
    """Based on the corpus, and given 'n', construct all lower N-gram models.
    Using the words in the sentence, use the n-gram model to predict the next words
    (upto 10 or till the first '.', '?', '!')
    """
    if n < 1:
        print("n must be greater than or equal to 1. ")
        return []

    if n > len(sentence):
        print("Input word list is lesser than the N-grams model. Input atleast N words")
        return []

    corpus = [
        word.strip()
        for word in corpus
        if word not in ("'", '"', '""', "''", "``", "`", "--", "-")
    ]
    n_gram_dict = defaultdict(defaultdict)
    uni_gram_deterministic = most_frequent(corpus)
    # Initialise the n-gram transition table
    current_n_gram = n
    while current_n_gram >= 1:
        current_n_gram = current_n_gram - 1
        n_gram_dict[current_n_gram] = {}
        for index in range(current_n_gram, len(corpus) - current_n_gram):
            # Avoid n-grams which overlap over two sentences
            # i.e. if any of ., ? or ! occur anywhere other than the first or last index
            if any(x in corpus[index - n + 1 : index - 1] for x in [".", "?", "!"]):
                continue

            # Increment or initialze the count of next work
            prev_words = "".join(corpus[index - current_n_gram : index])
            current_word = corpus[index]

            if not prev_words in n_gram_dict[current_n_gram].keys():
                n_gram_dict[current_n_gram][prev_words] = {}

            if not current_word in n_gram_dict[current_n_gram][prev_words].keys():
                n_gram_dict[current_n_gram][prev_words][current_word] = 0

            n_gram_dict[current_n_gram][prev_words][current_word] += 1

    print("The size of the dictionary is {} bytes".format(sys.getsizeof(n_gram_dict)))
    for key, value  in n_gram_dict.items():
        print(key)
        print(len(n_gram_dict[key]))
    current_n = n - 1
    while True:
        try:
            # Break condition
            if any(x in sentence for x in [".", "?", "!"]) or (len(sentence) >= 10):
                break

            # Depending on the value of the "deterministic" flag
            # Append the next predicted word to the list
            prediction_string = "".join(sentence[-current_n:])
            next_word_list = n_gram_dict[current_n][prediction_string]
            if deterministic:
                next_word = max(next_word_list, key=next_word_list.get)
                sentence.append(next_word)
            else:
                next_word = random.choice(list(next_word_list.keys()))
                sentence.append(next_word)
            current_n = n - 1
        except (ValueError, KeyError):
            current_n -= 1
            if current_n < 1:
                if deterministic:
                    sentence.append(uni_gram_deterministic)
                else:
                    sentence.append(random.choice(corpus))
            continue

    return sentence

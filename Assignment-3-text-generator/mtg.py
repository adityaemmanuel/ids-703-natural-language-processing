"""
Natural Language Processing - Assignment 2
Text Generation using N-gram Markov Text Generators
"""
from collections import defaultdict
import random


def finish_sentence(sentence, n, corpus, deterministic=False):
    """
    Based on the corpus, and given 'n', construct an N-gram model.
    Using the words in the sentence, use the n-gram model to predict the next words
    (upto 10 or till the first '.', '?', '!')
    """
    if n < 2:
        print("n must be greater than or equal to 2. ")
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
    n = n - 1

    # Initialise the n-gram transition table
    for index in range(n, len(corpus) - n):
        # Avoid n-grams which overlap over two sentences
        # i.e. if any of ., ? or ! occur anywhere other than the first or last index
        if any(x in corpus[index - n + 1 : index - 1] for x in [".", "?", "!"]):
            continue

        # Increment or initialze the count of next work
        prev_words = "".join(corpus[index - n : index])
        current_word = corpus[index]
        try:
            n_gram_dict[prev_words][current_word] += 1
        except KeyError:
            n_gram_dict[prev_words][current_word] = 1

    while True:
        try:
            # Break condition
            if any(x in sentence for x in [".", "?", "!"]) or (len(sentence) >= 10):
                break

            # Depending on the value of the "deterministic" flag
            # Append the next predicted word to the list
            prediction_string = "".join(sentence[-n:])
            next_word_list = n_gram_dict[prediction_string]
            if deterministic:
                next_word = max(next_word_list, key=next_word_list.get)
                sentence.append(next_word)
            else:
                next_word = random.choice(list(next_word_list.keys()))
                sentence.append(next_word)
        except ValueError:
            next_word = "<<UNKNOWN N-GRAM>>"
            sentence.append(next_word)

    return sentence

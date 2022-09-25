"""Markov Text Generator.

Patrick Wang, 2021

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""
import random
import nltk
from mtg import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    words = finish_sentence(
        ["she", "was", "not"],
        3,
        corpus,
        deterministic=True,
    )
    print(words)
    assert words == ["she", "was", "not", "in", "the", "world", "."] or words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ",",
        "and",
        "the",
        "two",
    ]

    # Additional test_cases
    sentences = [
        ["i", "fought", "well"],
        ["this", "is", "an", "example", "test"],
        ["to", "be", "or", "not", "to", "be"],
        ["luke", "I", "am", "your", "father"],
        ["wishful", "thinking"],
    ]
    n_values = [3, 5, 6, 4, 2]

    for sentence, n in zip(sentences, n_values):
        print("#" * 50)
        print("input: ", sentence, n)
        output = finish_sentence(sentence, n, corpus, random.choice([True, False]))

        print("output: ", output)


if __name__ == "__main__":
    test_generator()

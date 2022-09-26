"""Markov Text Generator.

Patrick Wang, 2021

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""
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
        ["i", "did", "not", "want"],
        ["i", "did", "not", "want"],
        ["they", "are", "going", "to", "do"],
        ["luke", "i", "am", "your", "father"],
        ["for", "now", "this", "will", "be", "not"],
        ["i", "said", "yolo"],
        ["i", "said", "yolo"]
    ]
    n_values = [2, 4, 5, 5, 6, 3, 3]
    deterministic_flag = [True, True, True, False, True, False, True]

    for sentence, n, flag in zip(sentences, n_values, deterministic_flag):
        print("#" * 50)
        print("input: ", sentence, n)
        output = finish_sentence(sentence, n, corpus, flag)

        print("output: ", output)


if __name__ == "__main__":
    test_generator()


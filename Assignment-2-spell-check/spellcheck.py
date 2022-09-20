"""
Assignment-2 Natural Language Processing
"""
import math
import numpy as np
import argparse

def create_word_list(file_location):
    """
    Coverts the given word corpus to a probability frequency
    """
    word_list_probability = {}
    with open(file_location, encoding="utf-8") as file:
        for line in file:
            word_list_probability[line.split()[0].lower().strip()] = int(
                line.split()[1]
            )

    total_word_count = sum(word_list_probability.values())
    word_list_probability = {
        key: word_count / total_word_count
        for key, word_count in word_list_probability.items()
    }
    return word_list_probability


def calculate_levenshtein_distance(word_1, word_2):
    """
    Calculates the levenshtein distance between the two given words
    """
    distance_matrix = np.zeros((len(word_1) + 1, len(word_2) + 1))
    for index in range(len(word_1) + 1):
        distance_matrix[index][0] = index

    for index in range(len(word_2) + 1):
        distance_matrix[0][index] = index

    for row in range(1, len(word_1) + 1):
        for col in range(1, len(word_2) + 1):
            if word_1[row - 1] == word_2[col - 1]:  # the characters are the same
                distance_matrix[row][col] = distance_matrix[row - 1][col - 1]
                continue
            upper_val = distance_matrix[row - 1][col] + 1
            left_val = distance_matrix[row][col - 1] + 1
            diag_val = distance_matrix[row - 1][col - 1] + 1
            min_distance = min(upper_val, left_val, diag_val)
            distance_matrix[row][col] = min_distance

    return distance_matrix[row][col]


def find_closest_distance_word(input_word, word_list, normalizing_factor):
    """
    Returns the word that has the closest distance
    """
    input_word = input_word.lower().strip()
    if input_word in word_list:
        print(f"{input_word} is correct. It is present in the corpus")
        return

    score_dict = {}
    for word, prior_probability in word_list.items():
        lev_distance = calculate_levenshtein_distance(input_word, word)
        score_dict[word] = (lev_distance) * math.log(normalizing_factor) + math.log(
            prior_probability
        )
    closest_word = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:1]
    print(f"{input_word} is misspelled. Corrected word is {closest_word[0][0]}")

def run_spellcheck(word, word_corpus_location, normalizing_factor):
    """
    Wrapper function that runs the script
    """
    word_list = create_word_list(word_corpus_location)
    find_closest_distance_word(word, word_list, normalizing_factor)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--word", help="Enter the word to be checked", default='apple')
    parser.add_argument("--corpus_location", help="Enter the corpous location (local .txt file)", default='wordlist_frequency.txt')
    parser.add_argument("--normalizing_factor", help="Enter the normalizing factor", default=.00001)
    
    args=parser.parse_args()
    run_spellcheck(args.word, args.corpus_location, args.normalizing_factor)

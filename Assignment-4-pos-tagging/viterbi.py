"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
import math
import numpy as np

def viterbi(observations, transmission_frequency, emission_frequency):
    # Intital variable set up
    possible_pos_states = []
    possible_pos_freq = []
    all_states = get_all_states(transmission_frequency)
    
    # Calculate the observation probabilities using Transmission and Emission frequencies
    for index, word in enumerate(observations):        
        word = word.lower()
        if index == 0: # For the initial state, calculate all the prior probabilities
            for state in all_states:
                state_frequency = np.log(transmission_frequency['START'][state]*emission_frequency[word][state])
                possible_pos_states.append([state])
                possible_pos_freq.append([state_frequency])
    
        else: # For following states, calculate the 'max' probabable state based on prior/prev state
            for index in range(len(possible_pos_states)):
                prev_state = possible_pos_states[index][-1]
                prev_freq = possible_pos_freq[index][0]
                max_freq = -math.inf
                for state in all_states:
                    temp_freq = np.log(transmission_frequency[prev_state][state]*emission_frequency[word][state])
                    if temp_freq*prev_freq > max_freq:
                        max_freq = temp_freq*prev_freq
                        max_state = state
                possible_pos_states[index].append(max_state)
                possible_pos_freq[index][0] = max_freq
    
    max_index = np.argmax(possible_pos_freq)
    return possible_pos_states[max_index]

def get_all_states(transmission_frequency):
    all_states = set()
    for key in transmission_frequency.keys():
        all_states.add(key)
        for key_ in transmission_frequency[key].keys():
            all_states.add(key_)
    return list(all_states)
'''
def viterbi(obs, pi, A, B):
    """Viterbi POS tagging."""
    n = len(obs)

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [0]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [np.empty((0,))] * n
    qs[-1] = np.argmax(log_d[-1])
    for i in range(n - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)
'''
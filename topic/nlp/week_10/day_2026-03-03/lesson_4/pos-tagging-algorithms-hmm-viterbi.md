---
title: "POS Tagging Algorithms (HMM, Viterbi)"
date: "2026-03-03"
week: 10
lesson: 4
slug: "pos-tagging-algorithms-hmm-viterbi"
---

# Topic: POS Tagging Algorithms (HMM, Viterbi)

## 1) Formal definition (what is it, and how can we use it?)

Part-of-Speech (POS) tagging is the process of assigning a grammatical tag (e.g., noun, verb, adjective, adverb) to each word in a sentence.  It's a crucial step in many NLP pipelines because it provides valuable syntactic information that can be used for tasks like parsing, information retrieval, and machine translation.

A Hidden Markov Model (HMM) is a probabilistic sequence model used to describe systems that transition between hidden states while emitting observable symbols. In the context of POS tagging:

*   **Hidden States:** The POS tags themselves (e.g., NN, VB, JJ). We assume the 'true' POS tag sequence is hidden.
*   **Observations:** The words in the sentence. We *observe* the words.
*   **Transition Probabilities:** The probability of transitioning from one POS tag to another (e.g., P(VB | NN) - the probability of a verb following a noun). These are learned from training data.
*   **Emission Probabilities:** The probability of a word being generated from a particular POS tag (e.g., P("run" | VB) - the probability of the word "run" being a verb). These are also learned from training data.

The Viterbi algorithm is a dynamic programming algorithm used to find the most likely sequence of hidden states (POS tags) given a sequence of observations (words) and the HMM parameters (transition and emission probabilities). It efficiently searches through all possible tag sequences to find the one with the highest probability.

**How we use it:** Given a sentence, we use the HMM (with trained transition and emission probabilities) and the Viterbi algorithm to predict the most probable sequence of POS tags for that sentence.

## 2) Application scenario

Consider the sentence: "The cat sat on the mat."

Without POS tagging, we might only know the individual words. With POS tagging, we can identify:

*   The: Determiner (DT)
*   cat: Noun (NN)
*   sat: Verb (VBD)
*   on: Preposition (IN)
*   the: Determiner (DT)
*   mat: Noun (NN)

This POS tagged sentence can be used for:

*   **Chunking/Shallow Parsing:**  Identifying noun phrases (e.g., "The cat", "the mat") and verb phrases (e.g., "sat on the mat").
*   **Information Extraction:** Extracting entities and relationships from text. For example, identifying the subject (cat) and the location (on the mat).
*   **Machine Translation:**  POS tags can help disambiguate words with multiple meanings (e.g., "run" as a verb vs. "run" as a noun) and ensure correct grammatical structure in the target language.
*   **Text-to-Speech (TTS):** Proper pronunciation can depend on the POS tag (e.g., "read" (verb) vs. "read" (noun)).
*   **Search Engines:**  Improve search results by considering the grammatical role of keywords.  Searching for "running shoes" is different than searching for "run in the race", which would have a verb phrase.

## 3) Python method (if possible)

While implementing the Viterbi algorithm and HMM from scratch is a good learning exercise, several Python libraries provide pre-trained POS taggers that utilize HMMs (and other techniques) under the hood.  One example is `nltk` (Natural Language Toolkit).

```python
import nltk

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download('averaged_perceptron_tagger')

text = "The cat sat on the mat."
tokens = nltk.word_tokenize(text)  # Tokenize the sentence

# Use the averaged_perceptron_tagger (often uses a variant of HMMs/Viterbi)
pos_tags = nltk.pos_tag(tokens)

print(pos_tags)

# Sample HMM implementation (simplified, for demonstration purposes only,
# requires significant data preparation and handling of unseen words for real use)
import numpy as np

class SimpleHMM:
    def __init__(self, states, observations, transition_probs, emission_probs):
        self.states = states
        self.observations = observations
        self.transition_probs = transition_probs  # state x state matrix
        self.emission_probs = emission_probs      # state x observation matrix

    def viterbi(self, obs):
        n_states = len(self.states)
        n_obs = len(obs)

        viterbi_matrix = np.zeros((n_states, n_obs))
        backpointer_matrix = np.zeros((n_states, n_obs), dtype=int)

        # Initialization (t = 0)
        for s in range(n_states):
            viterbi_matrix[s, 0] = self.transition_probs[0, s] * self.emission_probs[s, obs[0]]  # Initial state to s
            backpointer_matrix[s, 0] = 0

        # Recursion (t > 0)
        for t in range(1, n_obs):
            for s in range(n_states):
                max_prob = -1
                best_state = -1
                for prev_s in range(n_states):
                    prob = viterbi_matrix[prev_s, t-1] * self.transition_probs[prev_s, s] * self.emission_probs[s, obs[t]]
                    if prob > max_prob:
                        max_prob = prob
                        best_state = prev_s

                viterbi_matrix[s, t] = max_prob
                backpointer_matrix[s, t] = best_state

        # Termination
        best_path_prob = np.max(viterbi_matrix[:, n_obs-1])
        best_path_end = np.argmax(viterbi_matrix[:, n_obs-1])

        # Backtracking
        best_path = [best_path_end]
        for t in range(n_obs - 2, -1, -1):
            best_path.insert(0, backpointer_matrix[best_path[0], t+1])

        return [self.states[i] for i in best_path] #return state sequence (POS tag sequence)


# Example usage (extremely simplified; needs robust training data)
states = ['NN', 'VB'] #Noun, verb.
observations = ['cat', 'sat', 'mat']
#Assume that the transition probabilities have been learned previously
transition_probs = np.array([
    [0.0, 1.0],  # Start -> NN or VB
    [0.7, 0.3],  # NN -> NN or VB
    [0.4, 0.6]   # VB -> NN or VB
])

#Assume that the emission probabilities have been learned previously
emission_probs = np.array([
    [0.8, 0.0], # P(cat | NN) and P(sat | NN)
    [0.0, 0.7] # P(cat | VB) and P(sat | VB)
])
emission_probs = np.column_stack([emission_probs, [0.05, 0.0]]) #Add "mat"
#Ensure probabilities sum to 1 with a smoothing

# Map words to numerical indices
observation_indices = [0,1,2] # "cat sat mat"
#print(observation_indices)

# Initialize and run the HMM
hmm = SimpleHMM(states, observations, transition_probs, emission_probs)

# Run Viterbi, requires to input index of observation
predicted_tags = hmm.viterbi(observation_indices) # observation indices
print(f"Simple HMM predicted tags: {predicted_tags}") #Not properly initialized with start symbol.
```

**Explanation:**

1.  **`nltk.pos_tag()`:** This function from the `nltk` library provides a pre-trained POS tagger.  It uses an averaged perceptron tagger, which achieves good accuracy and is relatively efficient. It does *not* allow you to directly control the specific HMM parameters.
2. **Simple HMM**: Contains `states`, `observations`, `transition_probs`, and `emission_probs`. Implements the Viterbi algorithm. This HMM contains many assumption which would need to be considered in practice.

**Important Considerations:**

*   **Training Data:** The accuracy of the HMM depends heavily on the quality and quantity of the training data used to estimate the transition and emission probabilities.
*   **Unknown Words:**  The HMM needs a strategy for handling words that were not seen during training (e.g., using smoothing techniques or assigning them a default tag).
*   **Smoothing:** The training data might not include every possible word and tag combination, resulting in zero probabilities. Smoothing techniques (e.g., Laplace smoothing, add-k smoothing) are used to avoid zero probabilities and improve the model's robustness.
*   **Start and End Symbols:**  HMMs often use special "start" and "end" symbols to model the beginning and end of sentences.
*   **Scalability:** Viterbi Algorithm scales linearly with observation size O(n).
*   **Log Probabilities:** In practice, log probabilities are used to avoid underflow issues when multiplying many small probabilities.

## 4) Follow-up question

How do more advanced POS tagging algorithms, like those based on deep learning (e.g., BiLSTM-CRF), improve upon HMM-based approaches, particularly in handling long-range dependencies and rare words?
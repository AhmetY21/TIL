---
title: "Semantic Analogies with Word Vectors"
date: "2026-02-18"
week: 8
lesson: 5
slug: "semantic-analogies-with-word-vectors"
---

# Topic: Semantic Analogies with Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Semantic analogies with word vectors leverage the numerical representation of words learned by word embedding models (like Word2Vec, GloVe, or FastText) to identify and reason about relationships between words.  The core idea is that if vector spaces accurately reflect semantic relationships, then the vector difference between two words (e.g., "king" - "man") should be similar to the vector difference between another pair of words expressing a similar relationship (e.g., "queen" - "woman").

Formally, a semantic analogy question takes the form: "a is to b as c is to ?"  We seek to find a word 'd' such that the relationship between 'a' and 'b' is analogous to the relationship between 'c' and 'd'. Using word vectors, we can mathematically represent this as:

`vector(b) - vector(a) ≈ vector(d) - vector(c)`

To find the word 'd', we rearrange the equation to solve for `vector(d)`:

`vector(d) ≈ vector(b) - vector(a) + vector(c)`

We then search the vocabulary of the word embedding model for the word whose vector is closest (in terms of cosine similarity, or other distance metrics) to the calculated `vector(d)`. This closest word is then predicted as the answer to the analogy.

This technique can be used to explore and validate the semantic understanding captured by word embedding models.  A successful analogy completion demonstrates that the model has learned meaningful relationships between words. It can also be used to automatically generate analogies, explore different types of relationships, and even uncover biases present in the training data.

## 2) Application scenario

Consider a language learning application. We want to help users understand verb conjugations. We can use semantic analogies to provide examples. For instance:

* Input: "walk is to walking as talk is to ?"
*  Word Vector Calculation:  `vector("walking") - vector("walk") + vector("talk")`
*  The system finds the word vector closest to the result, which is ideally `vector("talking")`.
* Output: "talking"

This application allows users to implicitly learn grammatical rules by observing the relationships between word forms. Another scenario is in knowledge graph completion, where identifying missing relationships between entities can be formulated as an analogy problem. For example, "France is to Paris as Italy is to ?"  Solving this correctly would help complete knowledge graphs with missing capital cities.

## 3) Python method (if possible)

```python
import gensim
import numpy as np

# Load pre-trained Word2Vec model (or any other word embedding model)
# This assumes you have the model file 'GoogleNews-vectors-negative300.bin'
# in the same directory.  If not, download it from the internet or change the path.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def analogy(a, b, c, model):
    """
    Solves the analogy problem: a is to b as c is to ?
    Returns the word that best completes the analogy.
    """
    try:
        # Ensure all words are in the vocabulary
        if a not in model.key_to_index or b not in model.key_to_index or c not in model.key_to_index:
            print("One or more words not in vocabulary")
            return None

        vec_d_approx = model[b] - model[a] + model[c]
        best_word = model.most_similar(positive=[vec_d_approx], topn=1)[0][0]
        return best_word
    except KeyError as e:
        print(f"Error: {e}") #Handle cases where a word is not in the vocab.
        return None



# Example usage
result = analogy('man', 'king', 'woman', model)
print(f"man is to king as woman is to {result}")

result = analogy('france', 'paris', 'germany', model)
print(f"france is to paris as germany is to {result}")

result = analogy('walk', 'walking', 'talk', model)
print(f"walk is to walking as talk is to {result}")

#Example of a word not in the vocab
result = analogy('florp', 'king', 'woman', model)
print(result)

```

## 4) Follow-up question

How can we evaluate the performance of a word embedding model on semantic analogy tasks systematically? What metrics are commonly used, and what are the challenges in creating a comprehensive evaluation dataset?  Also, how do different word embedding models (Word2Vec, GloVe, FastText) typically perform on analogy tasks, and why might their performance differ?
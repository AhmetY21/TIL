## Topic: N-Grams and Language Models

**1- Provide formal definition, what is it and how can we use it?**

*   **N-Grams:** An N-gram is a contiguous sequence of *n* items (words, characters, syllables, etc.) from a given sequence of text or speech. For example, given the sentence "The quick brown fox jumps over the lazy dog", the 2-grams (or bigrams) would be: "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog". 3-grams (or trigrams) would be: "The quick brown", "quick brown fox", "brown fox jumps", etc. 1-grams are also called unigrams.

*   **Language Model (LM):** A language model is a probability distribution over sequences of words. Formally, it assigns a probability P(w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>m</sub>) to any sequence of *m* words. The probability essentially quantifies how "likely" that sequence is to occur in the language.

*   **N-gram Language Models:** These are a type of language model that uses the n-gram concept to approximate the probability of a word sequence. The core idea is to estimate the probability of a word appearing based on the *n-1* preceding words.  This is based on the Markov assumption: the probability of a word depends only on the preceding *n-1* words.  The key formula is:

    P(w<sub>i</sub> | w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>) = Count(w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>, w<sub>i</sub>) / Count(w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>)

    Where:

    *   w<sub>i</sub> is the i-th word in the sequence.
    *   Count(w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>, w<sub>i</sub>) is the number of times the n-gram (w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>, w<sub>i</sub>) appears in the training corpus.
    *   Count(w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>) is the number of times the (n-1)-gram (w<sub>i-n+1</sub>, ..., w<sub>i-1</sub>) appears in the training corpus.

*   **How we can use them:** N-gram language models are used for a variety of tasks:

    *   **Text generation:** Generating new text that resembles the style of the training data.
    *   **Machine Translation:** Evaluating the fluency of translated sentences.
    *   **Speech Recognition:** Improving the accuracy of speech-to-text systems by predicting the most likely word sequence.
    *   **Spelling Correction:** Suggesting corrections for misspelled words based on the surrounding context.
    *   **Authorship Attribution:** Identifying the author of a text based on their writing style (n-gram usage).
    *   **Sentiment Analysis:**  While less common now due to more advanced techniques, n-grams can contribute to sentiment analysis by identifying common phrases associated with positive or negative sentiment.
    *   **Autocompletion/Text Prediction:** Predicting the next word the user is likely to type.

**2- Provide an application scenario**

Application scenario: **Autocompletion in a search engine.**

Imagine a user starts typing a search query into a search engine like Google. As they type, the search engine tries to predict what the user is going to type next.

An N-gram language model can be used for this task.  The search engine has a vast corpus of previous search queries.  It trains an N-gram language model (e.g., a trigram model) on this corpus.

When the user types "best Italian...", the search engine would look up all trigrams starting with "best Italian" in its model. Based on the probabilities learned from the corpus, it could predict:

*   "best Italian restaurant" (high probability)
*   "best Italian food" (medium probability)
*   "best Italian recipes" (lower probability)

The search engine then displays these predictions to the user, allowing them to quickly select the desired search query, improving user experience and reducing typing effort.

**3- Provide a method to apply in python**

python
import nltk
from nltk.util import ngrams
from collections import defaultdict

def create_ngram_model(corpus, n):
    """
    Creates an n-gram language model from a corpus of text.

    Args:
        corpus: A list of sentences (strings).
        n: The order of the n-gram model (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        A dictionary representing the n-gram model. The keys are tuples representing 
        n-grams, and the values are their probabilities. Also returns a set of vocabulary.
    """
    model = defaultdict(lambda: defaultdict(lambda: 0))
    vocabulary = set()

    for sentence in corpus:
        sentence = sentence.lower() #Normalize case
        tokens = nltk.word_tokenize(sentence)
        vocabulary.update(tokens)
        for i in range(len(tokens)-n+1):
            ngram = tuple(tokens[i:i+n-1])
            next_word = tokens[i+n-1]
            model[ngram][next_word] += 1

    # Calculate probabilities
    for ngram in model:
        total_count = float(sum(model[ngram].values()))
        for word in model[ngram]:
            model[ngram][word] /= total_count

    return model, vocabulary


def predict_next_word(model, history, n, vocabulary):
    """
    Predicts the most likely next word given a history of n-1 words.

    Args:
        model: The n-gram language model.
        history: A tuple of n-1 words representing the history.
        n: The order of the n-gram model.
        vocabulary: A set containing the valid vocabulary

    Returns:
        The most likely next word (string), or None if no prediction can be made.
    """

    if history in model:
        possible_words = model[history]
        best_word = max(possible_words, key=possible_words.get)
        if best_word in vocabulary:
            return best_word
        else:
            return "UNK" #Token to replace unknown word. Not ideal solution.
    else:
        return None # No prediction available
# Example usage:
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown fox jumps over the lazy cat.",
    "The cat sleeps."
]

n = 3  # Trigram model
model, vocabulary = create_ngram_model(corpus, n)

history = ("the", "quick")
next_word = predict_next_word(model, history, n, vocabulary)

if next_word:
    print(f"Given history '{history}', the predicted next word is: {next_word}")
else:
    print(f"No prediction available for history '{history}'.")


**Explanation:**

1.  **`create_ngram_model(corpus, n)`:**
    *   Takes a list of sentences (`corpus`) and the n-gram order (`n`) as input.
    *   Tokenizes each sentence using `nltk.word_tokenize` (you'll need to `pip install nltk` and `nltk.download('punkt')` if you haven't already).
    *   Creates n-grams from the tokens using `ngrams` from `nltk.util`.
    *   Counts the occurrences of each n-gram and the words that follow it.
    *   Calculates the probabilities of each word given its preceding n-1 words.
    *   Returns a dictionary representing the model and a set representing the vocabulary.
2.  **`predict_next_word(model, history, n, vocabulary)`:**
    *   Takes the model, a history (tuple of n-1 words), the order, and vocabulary.
    *   Looks up the history in the model.
    *   If the history exists, it finds the word with the highest probability following that history.
    *   Returns the predicted word.

**Important Considerations:**

*   **Smoothing:** This code doesn't implement smoothing techniques (like Laplace smoothing or Kneser-Ney smoothing). Without smoothing, the model will assign zero probability to n-grams that weren't seen in the training data, leading to poor performance on unseen text. Smoothing is crucial for real-world applications.
*   **Out-of-Vocabulary (OOV) Words:** The code doesn't explicitly handle OOV words. It will treat words not seen during training as unknown. A common approach is to replace rare words with an `<UNK>` token during preprocessing.
*   **Tokenization:** The choice of tokenizer can significantly impact the performance of the model. Consider using a more sophisticated tokenizer if needed.

**4- Provide a follow up question about that topic**

How can we address the problem of *sparsity* in N-gram language models, especially when dealing with relatively small training datasets, and what are the common smoothing techniques used to mitigate this issue? Explain how one such smoothing technique works (e.g., Laplace smoothing or Kneser-Ney smoothing).
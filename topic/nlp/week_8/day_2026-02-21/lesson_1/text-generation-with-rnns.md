---
title: "Text Generation with RNNs"
date: "2026-02-21"
week: 8
lesson: 1
slug: "text-generation-with-rnns"
---

# Topic: Text Generation with RNNs

## 1) Formal definition (what is it, and how can we use it?)

Text generation with Recurrent Neural Networks (RNNs) is the task of using an RNN to automatically create new text based on a learned probability distribution from a training corpus. The core idea is to train the RNN to predict the next word (or character) given a sequence of preceding words (or characters).

More formally:

*   **What is it?**  We are trying to learn a probabilistic language model:  P(w<sub>t</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>t-1</sub>), where w<sub>t</sub> is the t-th word in a sequence and the model estimates the probability of observing the t-th word given all previous words.  RNNs are particularly well-suited for this task because they inherently maintain a hidden state that captures information about the sequence history.
*   **How can we use it?**
    *   **Language Modeling:**  As described above, learning the probability distribution of text.  This can be used in other NLP tasks like machine translation or speech recognition.
    *   **Text Generation:**  Generating novel text, such as:
        *   **Character-level generation:** Predicting the next character in a sequence.  This allows generating text letter by letter.
        *   **Word-level generation:** Predicting the next word in a sequence.
    *   **Creative Writing:**  Assisting in writing poems, scripts, or articles.
    *   **Code Generation:**  Generating code snippets.
    *   **Dialogue Generation:** Creating chatbots and conversational agents.

The basic process involves:
1.  **Data Preparation:**  Gather a large corpus of text.
2.  **Tokenization:** Convert the text into a sequence of tokens (words or characters).
3.  **Numericalization:**  Map each token to a numerical index.  This creates a vocabulary.
4.  **Model Training:** Train an RNN (e.g., LSTM or GRU) to predict the next token given a sequence of preceding tokens.  The RNN takes the numerical representation of the input sequence and outputs a probability distribution over the vocabulary.
5.  **Text Generation:**  Start with an initial seed sequence.  Feed this sequence into the trained RNN to predict the next token.  Sample a token from the predicted probability distribution (e.g., using techniques like greedy sampling or temperature sampling).  Append the sampled token to the sequence and repeat the process.

## 2) Application scenario

Imagine you want to build a chatbot that can generate responses in a style similar to a famous author, say Jane Austen.

1.  **Data Collection:** You would gather a large collection of Jane Austen's novels (e.g., "Pride and Prejudice," "Sense and Sensibility").
2.  **Preprocessing:**  Clean the text (remove punctuation, convert to lowercase).
3.  **Training:** You would train an RNN (LSTM or GRU would be suitable) on this data. The RNN learns the statistical patterns of Austen's writing style.
4.  **Chatbot Interaction:**
    *   A user provides an initial prompt, like "Mr. Darcy was quite..."
    *   The chatbot would encode this prompt and feed it to the trained RNN.
    *   The RNN would predict the next word with a certain probability distribution (e.g., "arrogant" - 0.4, "handsome" - 0.3, "reserved" - 0.1...).
    *   The chatbot might sample the word "arrogant" based on its probability.
    *   The chatbot would then append "arrogant" to the prompt, creating "Mr. Darcy was quite arrogant".
    *   The process repeats to generate a longer response, mimicking Austen's writing style.  For example, the chatbot might respond "Mr. Darcy was quite arrogant, but his virtues were undeniable."

## 3) Python method (if possible)

This example uses TensorFlow/Keras to demonstrate character-level text generation.

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
import numpy as np

# 1. Load and preprocess text data (replace with your actual data loading)
text = open('your_text_file.txt', 'r').read().lower()  # Replace with your file
chars = sorted(list(set(text)))
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. Create sequences and labels
seq_length = 100  # Length of input sequences
step = 3  # Step size to create overlapping sequences
sentences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

# 3. Vectorize the data
x = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# 4. Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(chars)))) # Input shape is (sequence_length, vocabulary_size)
model.add(Dense(len(chars), activation='softmax')) # Output a probability distribution over the vocabulary

# 5. Compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x, y, batch_size=128, epochs=10)

# 6. Text generation function
def generate_text(model, seed_text, length=400, temperature=0.5):
    generated_text = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        # Temperature sampling (adjust temperature for more or less randomness)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(probas)
        next_char = index_to_char[next_index]

        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    return generated_text

# 7. Generate text
seed_text = "the quick brown fox jumps over the lazy dog. " #Initial seed, needs to be at least seq_length
print(generate_text(model, seed_text, length=400))
```

**Explanation:**

1.  **Data Loading and Preprocessing:** Loads the text, converts it to lowercase, and creates a vocabulary of unique characters.  Also, creates mappings between characters and their indices.
2.  **Sequence Creation:** Creates overlapping sequences of `seq_length` characters from the text data. `next_chars` stores the character that follows each sequence.
3.  **Vectorization:** Converts the text sequences and the next characters into one-hot encoded vectors.  `x` represents the input sequences, and `y` represents the target (next) characters.
4.  **Model Building:** Builds an LSTM-based model. The model consists of an LSTM layer followed by a dense layer with a softmax activation function to output a probability distribution over the vocabulary.
5.  **Model Training:** Compiles and trains the model using categorical crossentropy loss and the Adam optimizer.  The `fit` method trains the model on the prepared data.
6.  **Text Generation Function:**  `generate_text` function takes the trained model, a seed text (initial sequence), and the desired length of the generated text as input.  It generates text by repeatedly predicting the next character based on the current sequence and sampling from the predicted probability distribution using a temperature parameter to control randomness. Lower temperatures make the output more deterministic, while higher temperatures introduce more randomness.
7.  **Text Generation:** Calls the `generate_text` function with a seed text to generate new text.

**Important Considerations:**

*   **Data Quality:** The quality and size of the training data significantly impact the generated text.
*   **Model Architecture:** Experiment with different RNN architectures (LSTM, GRU), number of layers, and hidden unit sizes.
*   **Hyperparameter Tuning:** Tune hyperparameters like learning rate, batch size, and number of epochs.
*   **Temperature:** The `temperature` parameter in the `generate_text` function controls the randomness of the generated text. Experiment with different temperature values.
*   **Vocabulary Size:** Choose between character-level or word-level tokenization based on your needs. Character-level models have smaller vocabularies but may require more training data. Word-level models can capture more semantic information.
*   **GPU:** Training RNNs, especially on large datasets, is computationally intensive. Consider using a GPU for faster training.

## 4) Follow-up question

How can attention mechanisms be incorporated into RNN-based text generation models to improve their performance, especially in capturing long-range dependencies within the text?
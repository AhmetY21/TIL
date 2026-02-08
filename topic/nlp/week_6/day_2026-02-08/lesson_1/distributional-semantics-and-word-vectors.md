## Topic: Distributional Semantics and Word Vectors

**1- Provide formal definition, what is it and how can we use it?**

*   **Distributional Semantics:** The core idea of distributional semantics is that the meaning of a word can be inferred from the words that frequently occur with it ("You shall know a word by the company it keeps," J.R. Firth). In other words, words that appear in similar contexts tend to have similar meanings. This is a data-driven approach that contrasts with symbolic approaches where meaning is defined using ontologies or logic.

*   **Word Vectors (Word Embeddings):** Word vectors are numerical representations of words that capture their semantic relationships based on the distributional hypothesis. These vectors are typically high-dimensional (e.g., 100-300 dimensions) and are learned from large corpora of text data. The dimensions themselves are not explicitly interpretable, but the relative positions of words in the vector space reflect their semantic similarity. For example, words like "king" and "queen" would be located closer together in the vector space than "king" and "apple".

*   **How can we use it?**
    *   **Semantic Similarity:** Compute the similarity between words or phrases based on the cosine similarity (or other distance metrics) of their corresponding word vectors. This can be used for tasks like finding synonyms or related terms.
    *   **Analogies:** Solve analogy problems like "a is to b as c is to ?" by performing vector arithmetic. For example, `vector("king") - vector("man") + vector("woman")` should result in a vector close to `vector("queen")`.
    *   **Text Classification:** Use word vectors as features for machine learning models to classify text documents based on their content. Word vectors can be aggregated (e.g., averaged or summed) to represent the meaning of a whole document.
    *   **Information Retrieval:** Improve search results by retrieving documents that contain words semantically related to the query terms, even if they don't contain the exact words in the query.
    *   **Machine Translation:** Align words across different languages by mapping their word vectors into a shared semantic space.

**2- Provide an application scenario**

**Scenario:** Sentiment analysis of product reviews.

**Application:** Imagine you want to automatically determine whether customer reviews for a new laptop are generally positive or negative. Instead of relying on simple keyword matching (e.g., counting the number of positive and negative words), you can use word vectors to capture the nuanced sentiment expressed in the reviews.

1.  **Word Vector Representation:** Each word in the review is represented by its corresponding word vector.
2.  **Review Representation:** The word vectors of all the words in a review are aggregated (e.g., averaged) to create a single vector representing the overall meaning of the review.
3.  **Sentiment Classification:** A machine learning classifier (e.g., logistic regression, support vector machine) is trained on a dataset of labeled reviews (positive, negative, neutral). The aggregated word vectors are used as features for the classifier.
4.  **Prediction:** Once trained, the classifier can predict the sentiment of new, unseen reviews based on their aggregated word vector representations.

This approach allows the sentiment analysis system to understand the sentiment expressed in the context of the entire review and to capture semantic relationships between words (e.g., recognizing that "great" is similar to "fantastic").

**3- Provide a method to apply in python**

python
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Sample corpus (replace with your own data)
sentences = [
    ["king", "is", "a", "strong", "man"],
    ["queen", "is", "a", "powerful", "woman"],
    ["man", "is", "mortal"],
    ["woman", "is", "mortal"],
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"]
]

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # vector_size is the dimensionality of the word vectors.

# Access the word vector for "king"
king_vector = model.wv["king"]
print("Vector for 'king':", king_vector)

# Calculate the similarity between "king" and "queen"
similarity = cosine_similarity([model.wv["king"]], [model.wv["queen"]])[0][0]
print("Similarity between 'king' and 'queen':", similarity)

# Find the most similar words to "king"
similar_words = model.wv.most_similar("king", topn=5)
print("Words similar to 'king':", similar_words)

# Example: Solving analogies
# The following finds the word which is closest to the vector
# model.wv["king"] - model.wv["man"] + model.wv["woman"]
# king - man + woman = ?
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("king - man + woman = ?: ", result)

#Save the model
model.save("word2vec.model")

#Load the model
loaded_model = Word2Vec.load("word2vec.model")
print("Loaded Model:", loaded_model)


**Explanation:**

1.  **Import Libraries:** Import `gensim` for Word2Vec and `sklearn` for cosine similarity.
2.  **Corpus:** `sentences` is a list of sentences, our training data. Replace this with your own corpus.
3.  **Word2Vec Model:** `Word2Vec` is initialized and trained on the corpus.
    *   `vector_size`: Defines the dimensionality of the word vectors.  A higher dimension can capture more nuanced relationships, but requires more data.
    *   `window`: The maximum distance between the current and predicted word within a sentence.
    *   `min_count`: Ignores all words with total frequency lower than this.
    *   `workers`: Use these many worker threads to train the model (=faster training with multicore machines).
4.  **Accessing Word Vectors:** `model.wv["king"]` retrieves the word vector for the word "king".
5.  **Cosine Similarity:** `cosine_similarity` calculates the cosine similarity between two word vectors. The closer the cosine similarity is to 1, the more similar the words are.
6.  **Most Similar Words:** `model.wv.most_similar` finds the words that are most similar to a given word based on their vector representations.
7.  **Analogy:** Demonstrates a simple analogy: king - man + woman ≈ queen.
8.  **Save and Load Model:** This example shows how to save a word2vec model and load it for future usage.

**4- Provide a follow up question about that topic**

How can we handle Out-Of-Vocabulary (OOV) words when using pre-trained word embeddings like Word2Vec or GloVe, especially if the OOV words are important for the task at hand (e.g., named entities in a news article)? What are the different strategies and their trade-offs?
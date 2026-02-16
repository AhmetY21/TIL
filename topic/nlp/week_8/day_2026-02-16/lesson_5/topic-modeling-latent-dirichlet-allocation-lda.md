---
title: "Topic Modeling: Latent Dirichlet Allocation (LDA)"
date: "2026-02-16"
week: 8
lesson: 5
slug: "topic-modeling-latent-dirichlet-allocation-lda"
---

# Topic: Topic Modeling: Latent Dirichlet Allocation (LDA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Dirichlet Allocation (LDA) is a generative probabilistic model for collections of discrete data such as text corpora. It's a form of unsupervised machine learning used to discover the abstract "topics" that occur in a collection of documents. LDA assumes documents are mixtures of topics, where each topic is a probability distribution over words.

Formally:

*   **Corpus:** A collection of documents.
*   **Document:** A sequence of words.
*   **Topic:** A distribution over words.
*   **Latent:** Hidden; LDA infers the topics without knowing them beforehand.
*   **Dirichlet:** A probability distribution over probability distributions. LDA uses Dirichlet distributions to model both the document-topic and topic-word distributions.

**Generative Process (simplified):**

1.  For each document:
    *   Choose a distribution over topics (using a Dirichlet distribution parameterized by alpha). This determines how much each topic contributes to the document.
2.  For each word in the document:
    *   Choose a topic from the document's topic distribution.
    *   Choose a word from the chosen topic's word distribution (using a Dirichlet distribution parameterized by beta).

**How can we use it?**

LDA can be used for:

*   **Topic Discovery:** Automatically discover the underlying topics present in a large collection of text.
*   **Document Classification/Categorization:** Assign documents to specific topics based on their topic distributions.
*   **Text Summarization:** Identify the key topics in a document and use them to create a concise summary.
*   **Information Retrieval:** Improve search engine results by understanding the topic content of documents.
*   **Exploratory Data Analysis:** Gain insights into the main themes and subjects present in a text dataset.

## 2) Application scenario

**Application Scenario: Analyzing Customer Reviews for Product Improvement**

Imagine a company wants to improve its new smartphone. They collect thousands of customer reviews from various online platforms. Manually reading and categorizing these reviews is time-consuming and impractical.

Using LDA, they can:

1.  **Preprocess the reviews:** Clean the text data by removing stop words, punctuation, and performing stemming or lemmatization.
2.  **Apply LDA:** Train an LDA model on the preprocessed review data. The model will discover latent topics discussed in the reviews.  For instance, LDA might identify topics such as:
    *   Topic 1: "Battery life" (words like battery, charge, long-lasting, drain)
    *   Topic 2: "Camera quality" (words like camera, image, quality, photos, resolution)
    *   Topic 3: "Screen display" (words like screen, display, brightness, resolution, colors)
    *   Topic 4: "Call quality" (words like call, audio, voice, reception, clear)

3.  **Analyze topic distributions:** Examine the proportion of each topic in different reviews. A review dominated by "Battery life" indicates the customer is primarily concerned about that aspect.
4.  **Identify problems and opportunities:** The company can identify common complaints (e.g., poor battery life) and positive feedback (e.g., excellent camera quality) based on the prevalence of these topics. They can then prioritize improvements based on the most frequently discussed topics and the sentiment associated with them. For example, if the "Battery life" topic is frequently associated with negative sentiment (using sentiment analysis), addressing battery life issues becomes a high priority.

## 3) Python method (if possible)

```python
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Sample documents (replace with your actual data)
documents = [
    "This is the first document about computer vision.",
    "This document is the second document related to machine learning.",
    "And this is the third one which is about computer vision",
    "Is this the first document about machine learning?",
    "The second document is not about either computer vision or machine learning, but about natural language processing."
]

# Preprocessing function
def preprocess_documents(documents):
    stop_words = set(stopwords.words('english'))
    processed_docs = []
    for doc in documents:
        # Tokenize the document
        word_tokens = word_tokenize(doc.lower()) #lower case
        # Remove stop words and non-alphabetic tokens
        filtered_words = [w for w in word_tokens if not w in stop_words and w.isalpha()]
        processed_docs.append(filtered_words)
    return processed_docs


processed_docs = preprocess_documents(documents)

# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(processed_docs)

# Filter out words that occur less than 2 documents, or more than 80% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.8)

# Convert document into the bag-of-words format = list of (token_id, token_count) tuples.
corpus = [dictionary.doc2bow(text) for text in processed_docs]


# Train the LDA model
num_topics = 3  # Number of topics to discover
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Print the topics and their corresponding words
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Get the topic distribution for a specific document (e.g., the first document)
doc_topics = lda_model.get_document_topics(corpus[0])
print("\nTopic distribution for the first document:", doc_topics)


# You can also save and load the model
# lda_model.save("lda_model.model")
# loaded_model = gensim.models.LdaModel.load("lda_model.model")

```

## 4) Follow-up question

How does the choice of the number of topics affect the interpretability and usefulness of the LDA model? What methods are there for choosing an optimal number of topics?
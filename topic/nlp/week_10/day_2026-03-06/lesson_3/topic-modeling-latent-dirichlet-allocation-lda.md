---
title: "Topic Modeling: Latent Dirichlet Allocation (LDA)"
date: "2026-03-06"
week: 10
lesson: 3
slug: "topic-modeling-latent-dirichlet-allocation-lda"
---

# Topic: Topic Modeling: Latent Dirichlet Allocation (LDA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Dirichlet Allocation (LDA) is a generative probabilistic model for collections of discrete data such as text corpora. In the context of NLP, LDA assumes that documents are mixtures of topics, where each topic is characterized by a distribution over words.  "Latent" refers to the fact that the topic structure is hidden and needs to be discovered.  "Dirichlet" refers to the type of prior distribution used for both the document-topic distributions and the topic-word distributions.

More specifically, LDA makes the following assumptions:

*   **Documents are mixtures of topics:** Each document is a probability distribution over topics.  For example, a document might be 70% about "politics" and 30% about "sports."
*   **Topics are distributions over words:** Each topic is a probability distribution over words. For example, the "politics" topic might have high probabilities for words like "election," "government," and "policy."

The generative process works as follows:

1.  For each topic *k* in *K* topics:
    *   Draw a distribution over words φ<sub>k</sub> ~ Dirichlet(β).  This is the topic-word distribution for topic *k*. β is a hyperparameter.

2.  For each document *d* in *D* documents:
    *   Draw a distribution over topics θ<sub>d</sub> ~ Dirichlet(α). This is the document-topic distribution for document *d*. α is a hyperparameter.
    *   For each word *w* in document *d*:
        *   Choose a topic z<sub>d,w</sub> ~ Multinomial(θ<sub>d</sub>). This is the topic assigned to word *w* in document *d*.
        *   Choose a word w<sub>d,w</sub> ~ Multinomial(φ<sub>z<sub>d,w</sub></sub>). This is the observed word.

We use LDA to infer the hidden topic structure of a corpus.  Given a collection of documents, LDA can identify the topics present in the corpus and the proportion of each topic in each document.  The parameters we want to estimate are φ (the topic-word distributions) and θ (the document-topic distributions).  We estimate these using techniques like Gibbs sampling or variational inference.

In summary, LDA is used to:

*   **Discover hidden topics:** Identify the key themes and concepts discussed in a collection of documents.
*   **Summarize documents:** Represent each document as a mixture of topics, providing a concise overview of its content.
*   **Classify documents:** Assign documents to specific topics based on their topic distributions.
*   **Explore data:**  Gain insights into the underlying structure of a corpus and uncover relationships between documents and topics.

## 2) Application scenario

Imagine you have a large collection of research papers in the field of computer science. Manually reading and categorizing all of them would be incredibly time-consuming. You can use LDA to automatically discover the main research areas being discussed in these papers.

*   **Input:** A corpus of computer science research papers.
*   **Application of LDA:** LDA will analyze the text of the papers and identify topics like "Deep Learning," "Computer Vision," "Natural Language Processing," "Database Systems," etc.  It will also estimate the proportion of each topic in each paper. For example, one paper might be 80% about "Deep Learning" and 20% about "Computer Vision."
*   **Output:** A list of topics, each characterized by a distribution of words (e.g., "Deep Learning" might be associated with words like "neural network," "backpropagation," "convolutional," "recurrent," etc.), and a topic distribution for each paper, indicating the proportion of each topic in that paper.

This allows you to quickly:

*   **Identify trending research areas:** By looking at the most prevalent topics.
*   **Find relevant papers:** By searching for papers with a high proportion of a specific topic.
*   **Understand the relationships between papers:** By comparing their topic distributions.
*   **Group papers:** Cluster papers by topic distribution, finding similar papers and simplifying browsing.

Another scenario could be analyzing customer reviews.  LDA can identify the topics that customers are discussing in their reviews, such as "product quality," "customer service," "shipping speed," etc.  This allows businesses to understand customer concerns and improve their products and services.

## 3) Python method (if possible)
```python
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Sample documents (replace with your actual data)
documents = [
    "This is the first document. It contains some words about machine learning.",
    "The second document is about natural language processing and text mining.",
    "This is the third document. Machine learning and data science are closely related.",
    "Natural language processing is a subfield of artificial intelligence.",
    "Text mining involves analyzing large amounts of text data to discover patterns."
]

# Preprocessing: Tokenization, stop word removal, and lowercasing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(document):
    tokens = word_tokenize(document.lower())
    tokens = [w for w in tokens if not w in stop_words and w.isalnum()] #remove punctuation and special chars
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# Create a dictionary (mapping between words and their integer ids)
dictionary = corpora.Dictionary(processed_docs)

# Create a corpus (term document frequency)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train the LDA model
num_topics = 2  # Specify the desired number of topics
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Print the topics and their top words
for topic_id, topic in lda_model.print_topics(-1):
    print(f"Topic {topic_id + 1}: {topic}")

# Get topic distribution for a specific document
doc_id = 0 # Choose the document index you want to analyze
topic_distribution = lda_model[corpus[doc_id]]
print(f"\nTopic distribution for document {doc_id + 1}: {topic_distribution}")

#To improve this code, you could add lemmatization or stemming.
```

## 4) Follow-up question

How can we evaluate the quality of the topics learned by an LDA model, and what are some common metrics used for this purpose? Can different hyperparameters (alpha and beta) influence these metrics?
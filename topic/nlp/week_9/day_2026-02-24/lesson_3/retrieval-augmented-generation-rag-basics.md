---
title: "Retrieval-Augmented Generation (RAG) Basics"
date: "2026-02-24"
week: 9
lesson: 3
slug: "retrieval-augmented-generation-rag-basics"
---

# Topic: Retrieval-Augmented Generation (RAG) Basics

## 1) Formal definition (what is it, and how can we use it?)

Retrieval-Augmented Generation (RAG) is a technique for enhancing the performance of large language models (LLMs) by grounding them with external knowledge retrieved from a data source. It addresses limitations such as hallucinations, limited knowledge cut-off dates, and inability to access up-to-date or proprietary information.

Here's a breakdown:

*   **Retrieval:** Instead of relying solely on its pre-trained parameters, the LLM first retrieves relevant information from an external knowledge source (e.g., a document database, a website, or a knowledge graph). This retrieval step is typically done using semantic search, keyword search, or a combination of both. The goal is to find pieces of information most relevant to the user's query.

*   **Augmentation:** The retrieved information is then incorporated into the prompt given to the LLM. This augmented prompt provides the LLM with the context needed to answer the user's question more accurately and comprehensively.

*   **Generation:** Finally, the LLM uses the augmented prompt to generate a response. Because the LLM has access to retrieved information, its response is more likely to be factually correct, relevant, and specific to the user's query.

In essence, RAG combines the generative capabilities of LLMs with the ability to access and incorporate external knowledge, creating a more powerful and versatile system. It can be used to answer questions, generate summaries, translate languages, write different kinds of creative content, and answer your questions in an informative way – all while grounding the output in verifiable data.

## 2) Application scenario

Consider a scenario where you want to build a chatbot that can answer questions about a company's internal policies and procedures. The company has a vast collection of documents outlining these policies, but the LLM itself doesn't have access to this information.

Using RAG, you can:

1.  **Create a vector database (embedding store) of your internal documents.** Each document is converted into a vector representation (embedding) that captures its semantic meaning. Tools like ChromaDB or Pinecone can be used for this.

2.  **When a user asks a question, perform a semantic search on the vector database.** The question itself is also converted into a vector embedding, and the system retrieves the documents whose vector embeddings are most similar to the question's embedding.

3.  **Augment the LLM prompt with the retrieved documents.** The prompt might include the user's question along with the relevant document snippets.

4.  **The LLM generates an answer based on the augmented prompt.** Because the LLM now has access to the relevant policy documents, it can provide an accurate and informative answer to the user's question.

This approach allows the chatbot to answer questions about internal policies even though the LLM was not specifically trained on that data. This avoids the need to retrain the LLM every time the policies change, as you only need to update the vector database.

## 3) Python method (if possible)

This example demonstrates a basic RAG implementation using LangChain and OpenAI. It requires you to have an OpenAI API key.

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


# 1. Load your data (example: a text file)
loader = TextLoader("my_company_policies.txt")  # Replace with your document path
documents = loader.load()

# 2. Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and store them in a vector database
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# 4. Create a retriever
retriever = db.as_retriever()

# 5. Create a QA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)


# 6. Ask a question
query = "What is the company's policy on remote work?"
answer = qa.run(query)

print(answer)
```

**Explanation:**

1.  **`TextLoader`**: Loads the document (`my_company_policies.txt`).
2.  **`CharacterTextSplitter`**: Splits the document into smaller chunks to improve retrieval relevance.  A common strategy.
3.  **`OpenAIEmbeddings`**: Creates embeddings of the text chunks using OpenAI's embedding model.
4.  **`Chroma`**:  Creates a Chroma vector database to store the embeddings.
5.  **`retriever`**:  Creates a retriever object from the Chroma database.
6.  **`RetrievalQA`**:  Creates a RetrievalQA chain, which combines the retriever and an LLM (OpenAI). The `chain_type="stuff"` tells LangChain to stuff all retrieved documents into the LLM's context window. Other chain types exist for handling larger sets of retrieved documents.
7.  **`qa.run(query)`**:  Runs the query against the QA chain, retrieving relevant documents, augmenting the prompt, and generating an answer.

**Note:**

*   You need to have the `langchain`, `openai`, `chromadb`, and `tiktoken` Python packages installed: `pip install langchain openai chromadb tiktoken`.
*   Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.
*   Replace `"my_company_policies.txt"` with the actual path to your document.
*   This is a basic example. For more complex scenarios, you might need to adjust the chunk size, the embedding model, the vector database, and the QA chain type.  Different chain types ('stuff', 'map_reduce', 'refine', 'map_rerank') handle different limitations such as exceeding the context window of the LLM.

## 4) Follow-up question

How can we improve the accuracy and relevance of the retrieved documents in a RAG system beyond simple keyword/similarity searches? What techniques can be employed to make the retrieval process more intelligent and context-aware?
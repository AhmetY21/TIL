---
title: "Retrieval-Augmented Generation (RAG) Basics"
date: "2026-03-13"
week: 11
lesson: 3
slug: "retrieval-augmented-generation-rag-basics"
---

# Topic: Retrieval-Augmented Generation (RAG) Basics

## 1) Formal definition (what is it, and how can we use it?)

Retrieval-Augmented Generation (RAG) is a natural language processing (NLP) technique that enhances the capabilities of large language models (LLMs) by grounding them in external knowledge sources. Instead of solely relying on the information encoded within their pre-trained parameters, RAG models first *retrieve* relevant information from a knowledge base and then *generate* a response based on both the retrieved context and their internal knowledge.

**What is it?** RAG combines information retrieval and text generation. It consists of two main components:

*   **Retriever:**  This component identifies and retrieves relevant documents or passages from an external knowledge source (e.g., a database, a collection of documents, a website).  The retriever uses techniques like vector similarity search to find information that is semantically related to the user's query.
*   **Generator:** This component is typically an LLM that takes the user's query *and* the retrieved context as input. It then generates a more informed and accurate response, leveraging both its pre-trained knowledge and the retrieved information.

**How can we use it?** RAG can be used to:

*   **Improve the accuracy of LLM responses:** By grounding the model in external knowledge, RAG reduces the risk of hallucinations (generating false or misleading information).
*   **Provide more up-to-date information:**  LLMs are typically trained on data from a specific point in time. RAG allows them to access and utilize more recent information from external sources.
*   **Customize responses to specific domains:** RAG enables LLMs to be adapted to particular industries or areas of expertise by retrieving information from domain-specific knowledge bases.
*   **Explain answers with source attribution:** RAG can provide references to the retrieved documents, increasing transparency and allowing users to verify the information.

## 2) Application scenario

Consider a scenario where you want to ask an LLM questions about a company's internal policies. The LLM has not been trained on this specific company's internal documentation. Without RAG, the LLM would likely provide generic answers or make up information.

With RAG, the process would look like this:

1.  **User asks:** "What is the company's policy on vacation time?"
2.  **Retriever:** The retriever searches a knowledge base of the company's policy documents (e.g., using vector similarity search). It retrieves relevant sections of the employee handbook that discuss vacation policies.
3.  **Generator:** The LLM receives the user's query *and* the retrieved text about vacation policies. It then generates a response that summarizes the policy, potentially including specific details like the number of vacation days allowed, the process for requesting time off, and any relevant restrictions.
4.  **Output:** "According to the employee handbook, full-time employees are eligible for 15 vacation days per year. Requests should be submitted to your manager at least two weeks in advance using the online portal. Vacation time cannot be carried over to the following year." (Potentially with a citation to the source document).

## 3) Python method (if possible)

This example uses Langchain, a popular framework for building applications with LLMs, and demonstrates a basic RAG implementation.  It utilizes `FAISS` for vector storage and `OpenAIEmbeddings` for creating document embeddings. Note that you will need an OpenAI API key and have Langchain and other necessary packages installed (`pip install langchain faiss-cpu tiktoken openai`).

```python
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set your OpenAI API key (or use environment variables)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Replace with your API key

# 1. Load your document(s)
loader = TextLoader("data/my_company_policy.txt") # Replace with your document path
documents = loader.load()

# 2. Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and store them in a vector database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# 4. Create a retriever
retriever = db.as_retriever()

# 5. Create a RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# 6. Ask a question
query = "What is the company's policy on sick leave?"
result = qa({"query": query})

# Print the answer and source documents
print("Answer:", result["result"])
print("\nSource Documents:")
for doc in result["source_documents"]:
    print(doc.metadata) # Print metadata, typically containing document source info.
    print(doc.page_content) # Print relevant retrieved snippets.
    print("---")
```

**Explanation:**

*   **Document Loading:** Loads the text from a file using `TextLoader`.  Replace `"data/my_company_policy.txt"` with the actual path to your policy document.
*   **Text Splitting:** Splits the document into smaller chunks using `CharacterTextSplitter`. This is crucial for efficient retrieval.
*   **Embedding Creation:** Creates embeddings of the text chunks using `OpenAIEmbeddings`. Embeddings are vector representations of the text, capturing semantic meaning.
*   **Vector Storage:** Stores the embeddings in a vector database (`FAISS`). This allows for fast similarity search.
*   **Retriever:** Creates a retriever from the vector database. The retriever is responsible for finding relevant text chunks based on a query.
*   **RetrievalQA Chain:** Creates a `RetrievalQA` chain, which combines the retriever and an LLM (`OpenAI`). The `stuff` chain type simply combines the retrieved documents into a single context string that's passed to the LLM along with the query.  `return_source_documents=True` is crucial to understand which portions of the knowledgebase led to the generated answer.
*   **Query and Answer:**  Asks a question and prints the answer, along with the source documents used to generate the answer.

## 4) Follow-up question

How can we evaluate the performance of a RAG system, especially in terms of retrieval quality and generation quality?  What metrics can we use, and what are some common challenges in RAG evaluation?
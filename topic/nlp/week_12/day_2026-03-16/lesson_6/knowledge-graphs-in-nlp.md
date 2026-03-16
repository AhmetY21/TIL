---
title: "Knowledge Graphs in NLP"
date: "2026-03-16"
week: 12
lesson: 6
slug: "knowledge-graphs-in-nlp"
---

# Topic: Knowledge Graphs in NLP

## 1) Formal definition (what is it, and how can we use it?)

A Knowledge Graph (KG) is a graph-structured representation of knowledge. It consists of entities (nodes) and relationships (edges) between those entities. Formally, it can be defined as a set of triples: (subject, predicate, object).

*   **Subject:**  The entity that the statement is about.  e.g., "Paris"
*   **Predicate:** The relationship or type of connection between the subject and object. e.g., "isCapitalOf"
*   **Object:**  The entity that the subject is related to. e.g., "France"

Therefore, the triple "(Paris, isCapitalOf, France)" represents the knowledge that Paris is the capital of France.

**How can we use KGs in NLP?**

Knowledge Graphs enhance NLP tasks by:

*   **Enriching Semantic Understanding:** KGs provide contextual information and background knowledge, improving the understanding of text.  For example, understanding the relationship between "apple" (the fruit) and "Apple" (the company) requires knowledge beyond the surface-level text.
*   **Improving Reasoning and Inference:**  By traversing the graph, NLP systems can infer new facts and relationships. For example, if we know "John worksFor Apple" and "Apple isHeadquarteredIn Cupertino," we can infer "John is located in Cupertino."
*   **Enhancing Entity Recognition and Disambiguation:**  KGs help to identify and disambiguate entities mentioned in text. If a text mentions "Michael Jordan," the KG can help determine whether it refers to the basketball player or another person with the same name.
*   **Supporting Question Answering:** KGs allow answering complex questions by retrieving relevant information and relationships from the graph. Instead of simply retrieving documents, the KG allows for structured retrieval of facts.
*   **Facilitating Text Generation:** KGs can guide text generation by providing a structured knowledge source to ensure coherence and factual correctness. For example, generating a biography can be guided by a KG containing information about the person.
*   **Improving Recommendation Systems:** KGs can represent user preferences and item characteristics, enabling more personalized recommendations.

## 2) Application scenario

**Application Scenario: Question Answering**

Consider a question: "Which actors starred in movies directed by Christopher Nolan?"

Without a KG, answering this question accurately requires complex semantic parsing and understanding of the relationships between actors, movies, and directors, followed by potentially lengthy document retrieval and reasoning.

Using a KG, we can:

1.  **Entity Recognition:** Identify "Christopher Nolan" as an entity in the KG.
2.  **Relationship Traversal:** Traverse the "directed" relationship from Christopher Nolan to his movies (e.g., Inception, The Dark Knight).
3.  **Relationship Traversal:** Traverse the "starredIn" relationship from each of these movies to the actors who starred in them (e.g., Leonardo DiCaprio, Christian Bale).
4.  **Answer Aggregation:**  Aggregate the list of actors obtained in step 3 to provide the final answer.

This approach provides a structured and efficient way to answer complex questions by leveraging pre-existing knowledge.  Large knowledge graphs like Wikidata or DBpedia are often used for such applications.

## 3) Python method (if possible)

We can use the `rdflib` library in Python to work with Knowledge Graphs, specifically RDF graphs, which are often used to represent KGs. This example demonstrates how to create a simple KG and query it.

```python
import rdflib
from rdflib import URIRef, Literal, Namespace

# Create a Graph
g = rdflib.Graph()

# Define Namespaces (optional, but good practice)
ex = Namespace("http://example.org/")
g.bind("ex", ex)  # Bind the namespace to a prefix

# Create Nodes (Entities)
alice = URIRef(ex.Alice)
bob = URIRef(ex.Bob)
movie1 = URIRef(ex.Movie1)

# Create Properties (Relationships)
knows = URIRef(ex.knows)
starredIn = URIRef(ex.starredIn)
title = URIRef("http://purl.org/dc/elements/1.1/title") # Using an existing standard namespace

# Add Triples (Statements) to the Graph
g.add((alice, knows, bob))
g.add((alice, starredIn, movie1))
g.add((bob, starredIn, movie1))
g.add((movie1, title, Literal("Awesome Movie")))

# Query the Graph
# Find everyone who knows Bob
for s, p, o in g.triples((None, knows, bob)):
    print(f"{s} knows {o}")

# Find the title of Movie1
for s, p, o in g.triples((movie1, title, None)):
    print(f"The title of {s} is {o}")

# Serialize the graph (optional)
# print(g.serialize(format="turtle"))
```

**Explanation:**

1.  **`rdflib.Graph()`:** Creates an RDF graph object.
2.  **`rdflib.URIRef()`:** Creates a Uniform Resource Identifier (URI) for entities and relationships.
3.  **`rdflib.Literal()`:**  Creates a literal value (e.g., a string).
4.  **`g.add()`:** Adds a triple (subject, predicate, object) to the graph.
5.  **`g.triples()`:**  Queries the graph for triples that match a given pattern.  `None` is used as a wildcard.
6.  **`g.serialize()`:**  Serializes the graph into a specific format (e.g., Turtle).

This example shows the basics of creating, populating, and querying a KG using `rdflib`.  More advanced operations include reasoning, inference, and integration with other NLP tools.

## 4) Follow-up question

How can Knowledge Graph Embeddings (KGEs) be used to improve the performance of NLP tasks, and what are some popular KGE techniques?
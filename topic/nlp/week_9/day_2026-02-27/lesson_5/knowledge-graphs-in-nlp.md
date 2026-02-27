---
title: "Knowledge Graphs in NLP"
date: "2026-02-27"
week: 9
lesson: 5
slug: "knowledge-graphs-in-nlp"
---

# Topic: Knowledge Graphs in NLP

## 1) Formal definition (what is it, and how can we use it?)

A Knowledge Graph (KG) is a structured representation of knowledge consisting of entities, concepts, and relationships between them.  Formally, a KG can be represented as a graph G = (V, E), where V is the set of vertices representing entities or concepts, and E is the set of edges representing relationships between these vertices. Each edge e ∈ E can be represented as a triple (head, relation, tail), often denoted as (h, r, t), where:

*   **h (head):** The source entity.
*   **r (relation):** The type of relationship between the entities (e.g., "is_a", "located_in", "authored_by").
*   **t (tail):** The target entity.

In the context of NLP, knowledge graphs are used to augment text understanding, enable reasoning, and improve the performance of various NLP tasks.  They provide a way to explicitly encode background knowledge and semantic relationships that are often implicit in text.

Here's how they're used:

*   **Semantic Understanding:** KGs can provide context and disambiguate word meanings. For instance, knowing "Apple" can be both a fruit and a company allows an NLP system to choose the correct meaning based on its relations within the KG.
*   **Reasoning:**  By traversing the KG, NLP systems can infer new information or answer complex questions.  For example, if the KG knows that "Steve Jobs" "founded" "Apple" and "Apple" "manufactures" "iPhone", the system can infer that "Steve Jobs" is related to "iPhone".
*   **Text Generation:** KGs can be used to guide the generation of more informative and coherent text by providing a structured source of information.
*   **Question Answering:** KGs can be used as a backend knowledge base to answer factual questions. A question is parsed, entities are identified, and then the KG is searched for relevant facts.
*   **Entity Linking/Recognition:**  KGs provide a place to link entities identified in text to canonical entries within the KG, resolving ambiguities and improving accuracy.
*   **Relationship Extraction:** KGs can be used as a training source for relation extraction models or as a validation check for extracted relations.

## 2) Application scenario

Consider the application of Question Answering.  A user asks the question: "Who directed the movie 'Pulp Fiction'?"

Without a knowledge graph, a standard NLP system might struggle if the information isn't explicitly present in the training data or requires complex text comprehension to infer.

Using a knowledge graph, the process would be:

1.  **Entity Recognition:** The system identifies "Pulp Fiction" as a movie entity.
2.  **Query Formation:**  The system formulates a query to the KG to find entities related to "Pulp Fiction" via the "directed_by" relation.
3.  **Knowledge Retrieval:**  The KG returns "Quentin Tarantino" as the director.
4.  **Answer Generation:** The system generates the answer: "Quentin Tarantino directed the movie Pulp Fiction."

In this scenario, the KG provides the factual knowledge needed to answer the question directly, without requiring the NLP system to deeply understand the nuances of the question or to extract the information solely from unstructured text. This is particularly valuable for questions involving complex relationships or uncommon facts.

## 3) Python method (if possible)

Using the `SPARQLWrapper` library to query a SPARQL endpoint of a knowledge graph (e.g., DBpedia).  This demonstrates how to access and retrieve information from a KG.

```python
from SPARQLWrapper import SPARQLWrapper, JSON

def query_dbpedia(query):
    """
    Queries DBpedia with the given SPARQL query.

    Args:
        query (str): The SPARQL query.

    Returns:
        list: A list of results, where each result is a dictionary.
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"Error querying DBpedia: {e}")
        return None

# Example: Find the director of Pulp Fiction
query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?directorLabel WHERE {
  dbr:Pulp_Fiction dbo:director ?director .
  ?director rdfs:label ?directorLabel .
  FILTER (lang(?directorLabel) = "en")
}
"""

results = query_dbpedia(query)

if results:
    for result in results:
        director_label = result["directorLabel"]["value"]
        print(f"Director: {director_label}")
else:
    print("No results found.")

```

**Explanation:**

1.  **`SPARQLWrapper`:** This library provides a convenient way to query SPARQL endpoints.
2.  **`query_dbpedia(query)` function:**
    *   Takes a SPARQL query as input.
    *   Sets the SPARQL endpoint to DBpedia.
    *   Sets the return format to JSON.
    *   Executes the query and converts the results to a Python dictionary.
    *   Handles potential errors during the query process.
3.  **SPARQL Query:** The example query is designed to find the director of the movie "Pulp Fiction" in DBpedia.  It uses prefixes to shorten the URIs for common namespaces like `rdfs` (RDF Schema) and `dbo` (DBpedia Ontology).  It selects the English label of the director.
4.  **Result Processing:** The code iterates through the results and prints the director's label.

This code snippet demonstrates a basic interaction with a knowledge graph using SPARQL, highlighting how you can retrieve information based on specific relationships and entities. This can be embedded within an NLP pipeline to augment text understanding or question answering capabilities.  DBpedia is a large, publicly available knowledge graph derived from Wikipedia.

## 4) Follow-up question

How can we automatically construct or update Knowledge Graphs from unstructured text data? What are the key challenges and techniques involved in Knowledge Graph Construction (KGC)?
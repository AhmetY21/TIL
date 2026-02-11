---
title: "Causal Graphs (DAGs) Basics"
date: "2026-02-11"
week: 7
lesson: 3
slug: "causal-graphs-dags-basics"
---

# Topic: Causal Graphs (DAGs) Basics

## 1) Formal definition (what is it, and how can we use it?)

A **Causal Graph**, specifically a **Directed Acyclic Graph (DAG)**, is a visual representation of causal relationships between variables.

*   **Directed:** The edges (arrows) between variables indicate the direction of the causal influence. An arrow from A to B (A → B) means that A *directly* causes B. A is often called the *parent* of B, and B is the *child* of A.

*   **Acyclic:** The graph contains no directed cycles. That is, you cannot start at a variable and follow the arrows to return to that same variable. This condition is crucial for representing causal relationships because cycles imply a variable causing itself, which is typically nonsensical in most real-world applications.

*   **Nodes:** The nodes in the graph represent variables. These variables can be observed or unobserved (latent).

**Uses:**

*   **Representing Causal Assumptions:**  DAGs explicitly encode our beliefs about how variables influence each other.  These assumptions are crucial for causal inference.  Without these assumptions, we cannot draw any causal conclusions from observational data.

*   **Identifying Causal Effects:** DAGs, along with causal identification techniques like the backdoor criterion, allow us to determine if we can estimate the causal effect of one variable on another from observational data, and if so, which variables need to be controlled for (conditioned on).

*   **Avoiding Collider Bias:** DAGs help us identify collider variables. A collider is a variable that is causally influenced by two or more other variables. Conditioning on a collider can induce spurious correlations between its parents, creating bias in causal effect estimation.

*   **Understanding Confounding:** DAGs illustrate confounding bias by showing common causes of both the treatment and the outcome.

In short, DAGs are powerful tools for thinking about causality and planning appropriate statistical analyses to estimate causal effects. They are essential for translating domain knowledge into a formal framework for causal inference.

## 2) Application scenario

Imagine we want to understand the effect of a new drug (D) on patient recovery (R).  We collect data on patients, including whether they received the drug (D), their age (A), and their overall health condition before treatment (H). We also observe their recovery status (R).

We suspect the following causal relationships:

*   Age (A) can influence both the likelihood of receiving the drug (D) and the recovery rate (R).
*   Overall health condition (H) also influences both the likelihood of receiving the drug (D) and the recovery rate (R).
*   The drug (D) directly affects recovery (R).

A DAG representing these relationships would look like this:

```
A --> D --> R
|         ^
v         |
H --------
```

In this scenario, `A` and `H` are confounders because they are common causes of both the treatment (D) and the outcome (R).  If we simply compare the recovery rates of patients who received the drug to those who didn't, we might be misled by the fact that older, sicker patients are less likely to recover *regardless* of the drug. Using the DAG and the backdoor criterion (or other identification techniques), we know we need to control for `A` and `H` to estimate the *causal* effect of `D` on `R`.

## 3) Python method (if possible)

While Python doesn't have a built-in library solely for creating and visualizing DAGs, several libraries are commonly used for this purpose in causal inference.  `networkx` is used to create the graph structure, and `matplotlib` or `graphviz` are used for visualization. `pgmpy` is a more specialized library providing functionality to handle probabilistic graphical models and DAGs, including causal inference methods.

Here's an example using `networkx` and `matplotlib`:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes (variables)
G.add_nodes_from(['A', 'D', 'H', 'R'])

# Add directed edges (causal relationships)
G.add_edges_from([('A', 'D'), ('A', 'R'), ('H', 'D'), ('H', 'R'), ('D', 'R')])

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # Position nodes for better visualization
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=15, font_weight="bold", arrowsize=20)
plt.title("Causal DAG")
plt.show()


# Example using pgmpy for a more direct DAG creation
from pgmpy.models import BayesianNetwork

# Define edges representing causal relationships
edges = [('A', 'D'), ('A', 'R'), ('H', 'D'), ('H', 'R'), ('D', 'R')]

# Create a BayesianNetwork (which is a type of DAG)
model = BayesianNetwork(edges)

# The networkx object is accessible via model.to_graph()
# Visualizing with networkx:
nx.draw(model.to_graph(), pos=nx.spring_layout(model.to_graph(), seed=42), with_labels=True, node_size=1500, node_color="lightgreen", font_size=15, font_weight="bold", arrowsize=20)
plt.title("Causal DAG using pgmpy")
plt.show()
```

This code snippet creates the DAG we described in the application scenario and displays it. While `networkx` creates the graph and `matplotlib` draws it, `pgmpy` handles the logic of identifying nodes, edges and potential downstream tasks.  `pgmpy` is more useful for performing causal inference tasks after creating the DAG. `graphviz` is another powerful option for visualization, especially for more complex DAGs, but it typically requires a separate installation.

## 4) Follow-up question

How do you deal with unobserved confounders in a DAG and what are the implications for causal inference? For example, consider the previous example, but suppose that the *true* health status is unobserved and only some proxies were observed. What effect might this have?
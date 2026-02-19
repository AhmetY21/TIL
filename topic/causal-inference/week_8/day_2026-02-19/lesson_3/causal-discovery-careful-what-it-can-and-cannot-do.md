---
title: "Causal Discovery (Careful: What it can and cannot do)"
date: "2026-02-19"
week: 8
lesson: 3
slug: "causal-discovery-careful-what-it-can-and-cannot-do"
---

# Topic: Causal Discovery (Careful: What it can and cannot do)

## 1) Formal definition (what is it, and how can we use it?)

Causal discovery refers to the process of learning causal relationships from observational data *without* relying on prior knowledge of the causal structure. It aims to infer a causal graph representing the probabilistic dependencies among variables, where an edge from variable A to variable B suggests that A causally influences B.  Unlike traditional statistical methods that primarily focus on correlations, causal discovery attempts to identify the *direction* of influence.

Formally, causal discovery algorithms aim to identify the underlying Directed Acyclic Graph (DAG) that best explains the observed data distribution.  A DAG consists of nodes representing variables and directed edges representing causal effects.  A crucial assumption underlying many causal discovery methods is the **Causal Markov Condition**, which states that a variable is independent of its non-descendants given its parents in the DAG.  Another important assumption is the **Faithfulness Assumption**, which states that all conditional independence relations in the data are entailed by the DAG; that is, there are no accidental independencies that would obscure the true causal structure.  Violation of these assumptions can lead to incorrect causal inferences.

Causal discovery can be used for various purposes:

*   **Hypothesis Generation:** Identify potential causal relationships that can be further investigated through experiments or interventions.
*   **Mechanism Understanding:** Gain insight into the underlying causal mechanisms that generate observed phenomena.
*   **Policy Evaluation:**  Predict the effects of interventions or policy changes on a system.
*   **Data Exploration:**  Reveal unexpected causal relationships that might be missed by traditional correlation analyses.

**Limitations (What it cannot do):**

*   **Cannot guarantee correctness:** Causal discovery methods are not infallible. The inferred causal graph is only as good as the data and the assumptions made by the algorithm. Violations of the Causal Markov Condition and/or the Faithfulness Assumption can lead to erroneous conclusions.
*   **Cannot distinguish between all DAGs:** Several DAGs can be Markov equivalent, meaning they encode the same conditional independence relationships. Causal discovery algorithms can typically only identify the Markov equivalence class, rather than a single DAG. This limitation means that directionality between some variable pairs might remain ambiguous.  These equivalence classes can often be represented using a CPDAG (Completed Partially Directed Acyclic Graph).
*   **Sensitive to data quality:** Noisy data, missing values, and small sample sizes can significantly impact the accuracy of causal discovery algorithms.
*   **Requires strong assumptions:**  As mentioned above, causal discovery relies on assumptions like the Causal Markov Condition and Faithfulness. These assumptions are often difficult to verify in practice.
*   **Difficult with latent variables:**  The presence of unobserved confounders (latent variables) can significantly complicate causal discovery. Many algorithms struggle to handle latent variables effectively, leading to biased causal inferences.
*   **Computational complexity:**  Causal discovery algorithms can be computationally expensive, especially for high-dimensional datasets.

## 2) Application scenario

**Scenario:**  Imagine a marketing team wants to understand the causal drivers of customer churn. They have collected data on various customer attributes, such as demographics, purchase history, website activity, and customer service interactions.

**Applying Causal Discovery:**

The team can use a causal discovery algorithm to identify which factors are most likely to *cause* customer churn. For example, the algorithm might reveal that frequent website visits combined with a low number of recent purchases causally lead to churn. This is more insightful than simply finding a correlation between website visits and churn because it suggests a potential intervention strategy: improve the purchasing experience for frequent website visitors.

**Benefits:**

*   **Targeted interventions:** Identify the most effective points for intervention to reduce churn.
*   **Reduced experimentation:**  Prioritize experiments based on the inferred causal relationships. Instead of testing every possible intervention, focus on those that are likely to have the greatest impact.
*   **Improved understanding:** Gain a deeper understanding of the underlying drivers of churn.

**Potential pitfalls:**

*   **Unobserved confounders:**  If there are unobserved factors that influence both customer attributes and churn (e.g., overall customer satisfaction with the product, which is not directly measured), the causal discovery algorithm might infer spurious relationships.
*   **Incorrect assumptions:**  If the Causal Markov Condition or Faithfulness Assumption is violated, the inferred causal graph might be incorrect. For example, a marketing campaign might accidentally create a near-independence between two variables that would otherwise be causally linked.

## 3) Python method (if possible)

One popular Python library for causal discovery is `causaldiscovery`.  Here's an example using the PC algorithm, a constraint-based method, which is implemented in the `cdt` (Causal Discovery Toolbox) package, which is part of the `causaldiscovery` umbrella package.

```python
# Requires installing the causaldiscovery package (which includes CDT)
# pip install causaldiscovery

import pandas as pd
import numpy as np
from causaldiscovery.cdt.causality.graph import PC

# Generate some synthetic data (replace with your actual data)
np.random.seed(0)
n_samples = 100
x = np.random.normal(0, 1, n_samples)
y = 2*x + np.random.normal(0, 1, n_samples)
z = 0.5*y + np.random.normal(0, 1, n_samples)
data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

# Initialize the PC algorithm
pc = PC(alpha=0.05)  # Significance level

# Run the algorithm
skeleton = pc.create_skeleton(data)
directed_graph = pc.orient_edges(skeleton, data)

# Print the inferred graph (adjacency matrix)
print(directed_graph.to_adjacency())

# You can also visualize the graph using networkx (requires installing networkx)
# import networkx as nx
# import matplotlib.pyplot as plt

# nx_graph = nx.DiGraph(directed_graph.to_adjacency())
# nx.draw(nx_graph, with_labels=True)
# plt.show()

```

**Explanation:**

1.  **Data Preparation:** The code first generates some synthetic data.  In a real application, you would replace this with your actual data loaded into a Pandas DataFrame.
2.  **Initialization:** The `PC` class is initialized with a significance level (`alpha`). This parameter controls the threshold for statistical independence tests.
3.  **Algorithm Execution:**
    *   `create_skeleton()`: This method constructs the undirected skeleton of the graph by performing conditional independence tests.
    *   `orient_edges()`: This method orients the edges of the skeleton to create a directed graph (DAG).
4.  **Output:** The inferred DAG is represented as an adjacency matrix.  The code also includes commented-out lines to visualize the graph using `networkx`.

**Important Notes:**

*   This is a simplified example. You may need to tune the parameters of the PC algorithm (e.g., `alpha`) and explore other causal discovery algorithms depending on your data and the assumptions you are willing to make.
*   The `causaldiscovery` package offers several other algorithms, including score-based methods (e.g., GES, LiNGAM) and constraint-based methods other than PC.
*   Always carefully consider the limitations of causal discovery and validate the inferred causal relationships with domain expertise and, ideally, through interventional experiments.
* Ensure that your data is appropriately pre-processed (e.g., handling missing values, scaling numerical features) before running the algorithm.

## 4) Follow-up question

How can we evaluate the performance of a causal discovery algorithm when we don't know the true causal graph? What metrics or approaches can be used to assess the quality of the inferred causal relationships?
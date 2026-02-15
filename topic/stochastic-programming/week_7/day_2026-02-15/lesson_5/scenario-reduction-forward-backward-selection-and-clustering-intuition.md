---
title: "Scenario Reduction: forward/backward selection and clustering intuition"
date: "2026-02-15"
week: 7
lesson: 5
slug: "scenario-reduction-forward-backward-selection-and-clustering-intuition"
---

# Topic: Scenario Reduction: forward/backward selection and clustering intuition

## 1) Formal definition (what is it, and how can we use it?)

Scenario reduction in stochastic programming is the process of simplifying a large, complex set of possible future scenarios (representing uncertainty) into a smaller, more manageable set. These scenarios are typically used to model uncertain parameters within an optimization problem. A high number of scenarios can lead to computationally intractable problems, while an inadequate number can lead to poor decisions. Scenario reduction aims to strike a balance between computational tractability and solution accuracy.

**Forward Selection:** Starts with an empty scenario set and iteratively adds the scenario that, when added to the current set, minimizes some distance measure (e.g., Wasserstein distance or Kantorovich distance) between the original scenario distribution and the reduced scenario distribution.

**Backward Selection:** Starts with the full scenario set and iteratively removes the scenario that, when removed from the current set, minimizes the increase in the distance measure between the original scenario distribution and the reduced scenario distribution.

**Clustering:** Groups similar scenarios into clusters and represents each cluster by a single representative scenario (e.g., the centroid of the cluster).  This reduces the number of scenarios while attempting to preserve the overall distribution of uncertainty. Common clustering algorithms include k-means and hierarchical clustering.

**How can we use it?**

*   **Computational Tractability:**  Reduces the size of stochastic programming problems, making them solvable within reasonable time and resource constraints.
*   **Model Simplification:** Makes the problem easier to understand and analyze.
*   **Robustness:** Selecting a representative subset of scenarios can help ensure that the solution is robust to a range of possible outcomes.
*   **Improved Solution Quality:** In some cases, scenario reduction can even lead to *better* solutions. This may seem counterintuitive, but sometimes, the noise introduced by a large number of closely similar scenarios can obscure the true optimal solution.
*   **Risk management:** Helps understand the range of possible outcomes and the potential risks associated with different decisions.

## 2) Application scenario

Consider a supply chain planning problem where the demand for a product is uncertain. We have historical data that represents 1000 possible demand scenarios for the next month. Directly incorporating all 1000 scenarios into a stochastic programming model to determine the optimal production and inventory levels would be computationally expensive.

Applying scenario reduction techniques, we can reduce the number of scenarios to a more manageable set, say 10 or 20, while preserving the essential characteristics of the demand uncertainty.

*   **Forward Selection:** We could start with an empty set and iteratively add the scenario that best represents the remaining scenarios based on a distance measure.
*   **Backward Selection:** We could start with all 1000 scenarios and iteratively remove the least important one, based on the chosen distance measure, until we have 10 or 20 scenarios remaining.
*   **Clustering (e.g., k-means):** We could group the 1000 scenarios into 10 clusters and represent each cluster by its centroid. The centroids would become the reduced set of scenarios, each representing a group of similar demand patterns.

The resulting stochastic programming model, using the reduced set of scenarios, would be much faster to solve. We can then use the solution (production and inventory levels) obtained from the simplified model to make real-world decisions. The choice of reduction method and number of scenarios involves a tradeoff between computational cost and solution quality (approximation error).

## 3) Python method (if possible)

```python
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def kmeans_scenario_reduction(scenarios, num_clusters):
    """
    Reduces a set of scenarios using k-means clustering.

    Args:
        scenarios (np.ndarray): A numpy array where each row represents a scenario.
        num_clusters (int): The desired number of clusters (reduced scenarios).

    Returns:
        np.ndarray: A numpy array containing the centroids of the clusters,
                    representing the reduced set of scenarios.
        np.ndarray: An array of weights, with each weight corresponding to a scenario
                   and reflecting the proportion of original scenarios represented
                   by the reduced scenario.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')  # n_init added
    kmeans.fit(scenarios)
    cluster_labels = kmeans.labels_ #Scenario to cluster assignment
    centroids = kmeans.cluster_centers_

    #Determine the weights of the reduced scenarios based on the number of elements in each cluster.
    weights = np.zeros(num_clusters)
    for i in range(num_clusters):
        weights[i] = np.sum(cluster_labels == i) / len(scenarios)

    return centroids, weights

# Example Usage:
if __name__ == '__main__':
    # Generate some sample scenarios (replace with your actual data)
    num_scenarios = 100
    scenario_dimension = 5
    scenarios = np.random.rand(num_scenarios, scenario_dimension)

    # Reduce the number of scenarios to 10 using k-means
    num_reduced_scenarios = 10
    reduced_scenarios, weights = kmeans_scenario_reduction(scenarios, num_reduced_scenarios)

    print("Original number of scenarios:", num_scenarios)
    print("Reduced number of scenarios:", num_reduced_scenarios)
    print("Shape of reduced scenarios:", reduced_scenarios.shape)  # Should be (10, 5)
    print("Weights:", weights)  # Sum should be 1
    print("Sum of weights:", np.sum(weights))
```

**Explanation:**

1.  **`kmeans_scenario_reduction(scenarios, num_clusters)` Function:**
    *   Takes the original scenarios (as a NumPy array) and the desired number of reduced scenarios as input.
    *   Uses the `KMeans` algorithm from `sklearn.cluster` to cluster the scenarios. `n_init='auto'` addresses a warning in newer versions of scikit-learn regarding the default value of `n_init`.
    *   Returns the cluster centroids (representing the reduced scenarios) and weights for each cluster (proportion of original scenarios represented).

2.  **Example Usage:**
    *   Generates sample scenarios using `np.random.rand()`.  **Replace this with your actual scenario data.**
    *   Calls the `kmeans_scenario_reduction` function to reduce the scenarios.
    *   Prints the shape of the reduced scenarios and their weights to verify the result.  The sum of weights should be close to 1.

**Important Notes:**

*   **Install `scikit-learn` and `numpy`:**  If you don't have them already, install them using `pip install scikit-learn numpy scipy`.
*   **Data Preprocessing:**  Consider scaling your scenario data (e.g., using `StandardScaler` from `sklearn.preprocessing`) if the different dimensions have vastly different scales.  This can improve the performance of k-means.
*   **Other Clustering Algorithms:** You can easily adapt this code to use other clustering algorithms (e.g., hierarchical clustering) by replacing the `KMeans` part with the appropriate code.
*   **Distance Measures:** The performance of scenario reduction methods heavily depends on the choice of the distance measure. The Euclidean distance, implicitly used by K-means, is a common choice, but other measures, such as the Wasserstein distance (also known as Earth Mover's Distance), may be more appropriate depending on the application. `scipy.stats` contains distance functions. You can use these with hierarchical clustering, for example.
* **Forward and Backward Selection:** Implementing forward and backward selection would require defining an appropriate distance metric between distributions of scenarios, and then iteratively adding or removing scenarios until the desired number is achieved. Libraries like `POT` (Python Optimal Transport) can facilitate calculation of Wasserstein distances, making the implementation of these methods more feasible.

## 4) Follow-up question

How does the choice of distance metric (e.g., Euclidean distance vs. Wasserstein distance) impact the performance and results of scenario reduction, and under what circumstances would one be preferred over the other? What are the computational complexities of each distance calculation and its effect on overall performance?
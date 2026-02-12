---
title: "Scenario Representation: sample space, scenarios, and probabilities"
date: "2026-02-12"
week: 7
lesson: 1
slug: "scenario-representation-sample-space-scenarios-and-probabilities"
---

# Topic: Scenario Representation: sample space, scenarios, and probabilities

## 1) Formal definition (what is it, and how can we use it?)

In stochastic programming, we deal with optimization problems where some parameters are uncertain. *Scenario representation* is the process of characterizing this uncertainty using a discrete set of possible outcomes. It involves defining:

*   **Sample Space (Ω):** The set of all possible realizations of the uncertain parameters. Ω is often very large or even infinite, making it impractical to work with directly.

*   **Scenarios (ω ∈ Ω):** A scenario is a specific realization of the uncertain parameters. Since we can't typically represent the entire sample space, we approximate it by selecting a finite set of scenarios, denoted as {ω₁, ω₂, ..., ωₙ}. Each scenario represents a specific state of the uncertain variables. The choice of scenarios is crucial for the accuracy of the stochastic programming model. For example, if we are modeling the demand for a product, each scenario might represent a particular level of demand.

*   **Probabilities (pᵢ):**  Each scenario ωᵢ is assigned a probability pᵢ, representing the likelihood of that scenario occurring.  The probabilities must sum to 1 (i.e., Σᵢ pᵢ = 1). These probabilities quantify our beliefs or estimates about the likelihood of each scenario.

**How we can use it:**

Scenario representation allows us to transform a stochastic optimization problem into a deterministic, but larger, problem. Instead of dealing with continuous probability distributions, we work with a finite set of possible outcomes. This allows us to:

*   Approximate the expected value function: We replace integrals over the continuous sample space with weighted sums over the discrete scenarios.
*   Formulate solvable optimization problems: We can create deterministic equivalent formulations of the stochastic problem, which can then be solved using standard optimization solvers.
*   Analyze the impact of uncertainty: By examining the solutions under different scenarios, we can understand how the optimal decisions are affected by the uncertain parameters.

## 2) Application scenario

Consider a supply chain problem where a company needs to decide how much product to order before knowing the actual demand. The demand is uncertain.

*   **Sample Space (Ω):**  The range of all possible demand values. This could be a continuous range of values (e.g., between 0 and infinity) or a discrete set of values (e.g., possible integer values).

*   **Scenarios (ω ∈ Ω):** Instead of trying to model the continuous distribution of demand, we create a few representative scenarios:
    *   ω₁: Low Demand (e.g., 100 units)
    *   ω₂: Medium Demand (e.g., 200 units)
    *   ω₃: High Demand (e.g., 300 units)

*   **Probabilities (pᵢ):** We assign probabilities to each scenario based on our understanding of the demand distribution:
    *   p₁ = 0.3 (30% chance of low demand)
    *   p₂ = 0.5 (50% chance of medium demand)
    *   p₃ = 0.2 (20% chance of high demand)

Now, the stochastic programming model can be formulated with these three scenarios, each with its corresponding probability. The optimization problem becomes:  "How much to order to minimize the expected cost, considering the cost of ordering, potential stockouts, and potential leftover inventory under each demand scenario?"

## 3) Python method (if possible)

While there isn't a single Python function to *create* scenarios (as this depends heavily on the problem), we can demonstrate how to *represent* them using dictionaries or Pandas DataFrames, which are common tools for working with data in Python for optimization.  We can also use NumPy arrays. Here's an example using a dictionary:

```python
import numpy as np

# Scenario representation using a dictionary
scenarios = {
    'low': {'demand': 100, 'probability': 0.3},
    'medium': {'demand': 200, 'probability': 0.5},
    'high': {'demand': 300, 'probability': 0.2}
}

# Example:  Calculate expected demand
expected_demand = 0
for scenario_name, scenario_data in scenarios.items():
    expected_demand += scenario_data['demand'] * scenario_data['probability']

print(f"Expected Demand: {expected_demand}")


# Scenario representation using NumPy arrays

demands = np.array([100, 200, 300])
probabilities = np.array([0.3, 0.5, 0.2])

expected_demand_np = np.sum(demands * probabilities)

print(f"Expected demand using numpy {expected_demand_np}")
```

This code demonstrates how to represent scenarios and their probabilities. The crucial part would be how these scenarios are integrated into an optimization model using a library like Pyomo or Gurobipy, where the scenario probabilities are used to formulate the objective function (usually minimizing expected cost or maximizing expected profit).

## 4) Follow-up question

How do you decide on the number and nature of the scenarios to use in a stochastic programming model?  Are there any techniques for scenario reduction or generation that can help improve the efficiency and accuracy of the model?
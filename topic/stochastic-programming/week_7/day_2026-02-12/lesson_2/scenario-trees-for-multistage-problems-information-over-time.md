---
title: "Scenario Trees for Multistage Problems (information over time)"
date: "2026-02-12"
week: 7
lesson: 2
slug: "scenario-trees-for-multistage-problems-information-over-time"
---

# Topic: Scenario Trees for Multistage Problems (information over time)

## 1) Formal definition (what is it, and how can we use it?)

A scenario tree is a discrete representation of uncertainty evolving over time, used in multistage stochastic programming problems. It's a directed tree graph where:

*   **Nodes:** Each node represents a possible state of the world at a particular time stage. The root node represents the initial state at time 0.
*   **Edges:** Each edge represents a possible realization of the uncertain parameters between two consecutive time stages. Edges are associated with a probability representing the likelihood of that realization occurring.
*   **Paths:** A path from the root to a leaf node (a node at the final time stage) represents a complete scenario, or a specific sequence of realizations of the uncertain parameters over the entire planning horizon.
*   **Stages:** The tree is structured into stages, corresponding to decision epochs. At each stage, a decision must be made, knowing only the information available up to that stage.

**How we use it:**

Scenario trees allow us to model multistage decision problems under uncertainty. We formulate optimization problems where decisions at each stage can depend on the scenario (the path from the root to the current node). The objective function typically includes a weighted average of costs/profits across all scenarios, where the weights are the probabilities of the scenarios.  The optimization problem is then solved to find a set of decisions that are "best" in expectation, considering the possible future uncertainties. Specifically:

*   **Representing Uncertainty:** They approximate the continuous distribution of uncertain parameters with a finite set of discrete scenarios.
*   **Decision-Making:**  They allow us to incorporate the impact of future uncertainties on current decisions. Decisions made in earlier stages must be robust enough to perform well under different possible future scenarios.
*   **Problem Formulation:**  They provide a framework for formulating multistage stochastic programming problems mathematically (e.g., using linear programming, mixed-integer programming). The decisions at each node are variables in the optimization problem, and the constraints link these decisions across stages and scenarios.

The quality of the solution depends heavily on the size and structure of the scenario tree.  Larger trees can represent the uncertainty more accurately but lead to computationally more demanding optimization problems. Scenario reduction techniques are often used to reduce the size of the tree while preserving its relevant statistical properties.

## 2) Application scenario

**Example: Supply Chain Management under Demand Uncertainty**

Imagine a company managing a supply chain over three months.  The demand for their product is uncertain each month.  They need to decide how much product to order at the beginning of each month.

1.  **Uncertainty:** The uncertain parameter is the demand for the product in each month.
2.  **Decision Stages:** The decision stages are the beginnings of each month (Month 1, Month 2, Month 3).
3.  **Scenario Tree:** We can construct a scenario tree to represent the possible demand scenarios:

    *   **Root Node (Month 0):** Initial state (e.g., initial inventory).
    *   **Month 1 Nodes:**  Two or three possible demand levels for Month 1 (e.g., Low, Medium, High). Each branch from the root represents a possible demand realization with an associated probability.
    *   **Month 2 Nodes:**  For each Month 1 node, branch out with possible demand levels for Month 2 (again, Low, Medium, High, and associated probabilities).
    *   **Month 3 Nodes:**  Similarly, for each Month 2 node, branch out with possible demand levels for Month 3.

4.  **Optimization:**  The stochastic programming problem would involve:

    *   **Decision Variables:** The quantity of product to order at the beginning of each month for each node in the scenario tree.
    *   **Objective Function:** Minimize the total expected cost, including ordering costs, inventory holding costs, and potential shortage costs (weighted by the scenario probabilities).
    *   **Constraints:**  Inventory balance equations (inventory at the end of the month depends on the initial inventory, orders, and demand). Constraints to avoid backorders, capacity limitations, etc.
5.  **Solution:** The solution to the stochastic program provides the optimal ordering quantities for each month under each demand scenario. This provides a more robust and informed decision than simply using a single forecast of demand.

## 3) Python method (if possible)

There isn't a single, built-in Python library dedicated solely to building and manipulating scenario trees. However, you can use general optimization libraries like Pyomo or Gurobi (with its Python API) along with data structures to represent the tree. Below is a simplified example using Pyomo to illustrate the concept of creating the tree structure and setting up the stochastic programming problem. This example focuses on *defining* the tree structure. Setting up the actual optimization model is a separate, more complex task.

```python
import pyomo.environ as pyo

# Define the number of stages and scenarios
num_stages = 3
num_scenarios_per_stage = 2 # Binary branching

# Create a dictionary to represent the scenario tree
scenario_tree = {}

# Root node
scenario_tree[0] = {0: {'probability': 1.0, 'children': {}}}

# Build the tree recursively
def build_tree(stage, parent_node, parent_scenario_id):
  if stage < num_stages:
    for i in range(num_scenarios_per_stage):
      scenario_id = parent_scenario_id * num_scenarios_per_stage + i + 1
      probability = 1.0 / num_scenarios_per_stage # Assuming equal probabilities for simplicity
      if stage+1 not in scenario_tree:
        scenario_tree[stage+1] = {}
      scenario_tree[stage+1][scenario_id] = {'probability': probability, 'children': {}}
      scenario_tree[stage][parent_node]['children'][i] = scenario_id # store children id to each node
      build_tree(stage + 1, scenario_id, scenario_id * num_scenarios_per_stage if stage < num_stages-1 else scenario_id)  # recursive call



build_tree(0, 0, 0)


# Print the scenario tree structure (simplified)
for stage in scenario_tree:
  print(f"Stage: {stage}")
  for node in scenario_tree[stage]:
    print(f"  Node: {node}, Probability: {scenario_tree[stage][node]['probability']}, Children: {scenario_tree[stage][node]['children']}")


# Example Usage within a Pyomo Model:
model = pyo.ConcreteModel()

#Sets
model.Stages = pyo.Set(initialize=range(num_stages+1))
model.Nodes = pyo.Set(initialize = [node for stage in scenario_tree for node in scenario_tree[stage]])
# Example Parameter (this is where you'd access the scenario-specific information)
model.Demand = pyo.Param(model.Nodes, initialize = {node: node%3+1 for node in model.Nodes}) #example, make it depend on actual scenario.

# Variables
model.Order = pyo.Var(model.Nodes, within = pyo.NonNegativeReals)

#Objective, Constraints will need to access tree structure and its stages /Nodes.

```

**Explanation:**

1.  **Tree Structure:** The code creates a dictionary `scenario_tree` to represent the tree. Each key in the dictionary is a stage number. Within each stage, the keys are the node IDs. Each node has a probability and a dictionary of children nodes.
2.  **Recursive Building:**  The `build_tree` function recursively creates the tree structure. You'll need to adapt this to your specific problem, incorporating how the uncertain parameters evolve between stages and their associated probabilities. In practice, these probabilities will probably come from simulation output or statistical modelling.
3.  **Pyomo Integration:** The example demonstrates how to define sets of Stages and Nodes for your Pyomo model.  The crucial part is using the scenario tree data structure within your Pyomo model to define scenario-dependent parameters, decision variables, and constraints. The `model.Demand` shows the usage of scenarios.

**Important Notes:**

*   This code only creates the tree structure. You still need to define the objective function, constraints, and solve the stochastic programming problem using Pyomo.
*   Real-world problems often require scenario reduction techniques to manage the computational complexity of large scenario trees. There are Python libraries available for scenario reduction (e.g., `scengen` - although it requires GAMS).
*   The probabilities used in the example are simplified. In practice, you would estimate them from data or using expert knowledge.
*   This is a simplified example and doesn't show how to create nonanticipativity constraints (ensuring that decisions are the same up to a certain stage for scenarios that share the same history).  That is a crucial aspect of implementing multistage stochastic programming models.

## 4) Follow-up question

How do you incorporate nonanticipativity constraints into a Pyomo model that uses a scenario tree? Provide a concise code snippet illustrating the implementation of such constraints.
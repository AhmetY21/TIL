---
title: "First-stage vs Second-stage Decisions (Here-and-now vs Wait-and-see)"
date: "2026-02-11"
week: 7
lesson: 6
slug: "first-stage-vs-second-stage-decisions-here-and-now-vs-wait-and-see"
---

# Topic: First-stage vs Second-stage Decisions (Here-and-now vs Wait-and-see)

## 1) Formal definition (what is it, and how can we use it?)

In stochastic programming, we explicitly acknowledge uncertainty in the parameters of an optimization problem.  We model this uncertainty through scenarios, probability distributions, or other representations of possible future events.  The core concept of first-stage vs. second-stage (or "here-and-now" vs. "wait-and-see") decisions concerns the timing and dependency of decisions in the face of this uncertainty.

*   **First-stage decisions (Here-and-now):** These decisions must be made *before* we observe the realization of the uncertain parameters (the scenario unfolds). They are often referred to as "strategic" decisions.  They must be robust in the sense that they should perform reasonably well across all possible scenarios. These decisions are typically represented by variables `x`.

*   **Second-stage decisions (Wait-and-see):** These decisions can be made *after* we observe the realization of the uncertain parameters. They are often referred to as "tactical" or "operational" decisions. They are scenario-dependent, meaning that their values depend on which scenario actually occurs. These decisions are typically represented by variables `y_s`, where `s` denotes a specific scenario.  Because they can adapt to the realized scenario, they provide a mechanism to hedge against the uncertainty inherent in the first-stage decisions.

**How can we use it?** This framework is fundamental to formulating stochastic programming problems. We use it by:

1.  **Identifying the uncertainty:** Characterize the uncertain parameters and their probability distribution (or other representation of uncertainty, such as scenario sets).
2.  **Defining the decision stages:** Determine which decisions must be made before (first-stage) and after (second-stage) the uncertainty is revealed.
3.  **Formulating the objective function:**  Typically, the objective function involves minimizing the expected cost (or maximizing the expected profit) over all scenarios. This includes the cost associated with first-stage decisions and the expected cost associated with the second-stage (scenario-dependent) decisions.
4.  **Formulating the constraints:**  Constraints are categorized based on their timing:
    *   **First-stage constraints:** Constraints that must hold regardless of the scenario. They involve only first-stage variables (`x`).
    *   **Second-stage constraints:** Constraints that must hold for each scenario. They involve both first-stage variables (`x`) and scenario-dependent second-stage variables (`y_s`).  These ensure feasibility given a particular outcome of the uncertain parameters.
5.  **Nonanticipativity:** In some stochastic programming formulations, we enforce nonanticipativity. This means that the first-stage decisions must be the same across all scenarios. This is often implicitly enforced by defining the first-stage variables outside the scenario loop in a mathematical programming model.

## 2) Application scenario

**Example:** Consider a supply chain network design problem.

*   **First-stage decision (Here-and-now):** Deciding where to build distribution centers (DCs). The number, size, and location of DCs are determined *before* we know the exact future demand.  Let `x_j = 1` if a DC is built at location `j`, and `0` otherwise.

*   **Uncertainty:** Future demand at customer locations. We can model this uncertainty with several demand scenarios, each with an associated probability.  Let `d_{i,s}` be the demand at customer location `i` in scenario `s`.

*   **Second-stage decision (Wait-and-see):**  Deciding how much to ship from each DC to each customer in each scenario.  These decisions are made *after* we observe the realized demand scenario.  Let `y_{j,i,s}` be the amount shipped from DC `j` to customer `i` in scenario `s`.

*   **Objective:** Minimize the total cost, which includes the fixed cost of building the DCs (first-stage cost) plus the expected transportation cost (second-stage cost) across all demand scenarios.

*   **Constraints:**
    *   First-stage constraints: Budget constraint on the total investment in building DCs.
    *   Second-stage constraints:
        *   Supply constraints at each DC (the amount shipped cannot exceed the DC's capacity).
        *   Demand constraints at each customer location (the demand must be satisfied).
        *   Non-negativity constraints on the shipping quantities.

In this scenario, the here-and-now decision (DC locations) is strategic, influencing the long-term structure of the supply chain. The wait-and-see decision (shipping quantities) is tactical, allowing the supply chain to adapt to the realized demand, mitigating the risk associated with uncertain demand.

## 3) Python method (if possible)

We can use Pyomo, a Python-based optimization modeling language, to formulate and solve stochastic programming problems.  Here's a simplified example illustrating the first-stage/second-stage concept:

```python
from pyomo.environ import *
import random

# Define the number of scenarios
num_scenarios = 3

# Define probabilities for each scenario
probabilities = {1: 0.3, 2: 0.4, 3: 0.3}

# Define uncertain parameters (e.g., demand) - this is just an example
demands = {
    1: 10 + random.randint(-3, 3),  # Scenario 1 demand
    2: 15 + random.randint(-3, 3),  # Scenario 2 demand
    3: 20 + random.randint(-3, 3)   # Scenario 3 demand
}

# Create the model
model = ConcreteModel()

# First-stage variable (here-and-now): Production quantity
model.x = Var(domain=NonNegativeReals)

# Define Scenarios
model.scenarios = Set(initialize=range(1, num_scenarios + 1))

# Second-stage variable (wait-and-see): Inventory level for each scenario
model.y = Var(model.scenarios, domain=NonNegativeReals)

# Objective function: Minimize production cost + expected inventory cost
def objective_rule(model):
    production_cost = model.x  # Cost per unit produced is 1
    expected_inventory_cost = sum(probabilities[s] * model.y[s] for s in model.scenarios)
    return production_cost + expected_inventory_cost
model.objective = Objective(rule=objective_rule, sense=minimize)

# Constraint:  Production must be sufficient to meet demand in each scenario
def constraint_rule(model, s):
    return model.x + model.y[s] >= demands[s]
model.constraint = Constraint(model.scenarios, rule=constraint_rule)


# Solve the model (assuming a solver like GLPK or CBC is installed)
solver = SolverFactory('glpk') # You may need to adjust the solver name
results = solver.solve(model)

# Print the results
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found:")
    print("Production quantity (first-stage):", model.x.value)
    for s in model.scenarios:
        print(f"Inventory level in scenario {s} (second-stage):", model.y[s].value)
else:
    print("Solver did not find an optimal solution.")

```

**Explanation:**

*   We define the number of scenarios and their probabilities.
*   `model.x` is the first-stage variable (production quantity) – decided *before* knowing the demand.
*   `model.scenarios` defines the possible scenarios.
*   `model.y[s]` is the second-stage variable (inventory level for each scenario) – decided *after* knowing the demand in scenario `s`.
*   The objective function minimizes the production cost plus the expected inventory cost across all scenarios.
*   The constraint ensures that production plus inventory is sufficient to meet demand *in each scenario*.  This links the first-stage and second-stage decisions.

This simplified example demonstrates the basic structure. Real-world problems will be more complex, involving more variables, constraints, and scenarios.

## 4) Follow-up question

How does the computational complexity of solving a stochastic programming problem with first-stage and second-stage decisions scale with the number of scenarios? Why is scenario reduction a common technique, and what are some approaches to scenario reduction?
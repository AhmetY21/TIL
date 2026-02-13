---
title: "Deterministic Equivalent Formulation (DEF) for two-stage SP"
date: "2026-02-13"
week: 7
lesson: 3
slug: "deterministic-equivalent-formulation-def-for-two-stage-sp"
---

# Topic: Deterministic Equivalent Formulation (DEF) for two-stage SP

## 1) Formal definition (what is it, and how can we use it?)

The Deterministic Equivalent Formulation (DEF) is a single, large-scale deterministic optimization problem that is equivalent to a two-stage stochastic programming problem. In essence, it unfolds the decision-making process across all possible realizations (scenarios) of the uncertain parameters.

Here's a breakdown:

*   **Two-Stage Stochastic Program:** A two-stage stochastic program involves making decisions in two stages.
    *   *First-stage decisions (here-and-now decisions):*  These decisions must be made *before* the uncertainty is revealed.  These are usually represented by the variable vector `x`.
    *   *Second-stage decisions (recourse decisions):* These decisions are made *after* the uncertainty is revealed. They are adjustments made to mitigate the consequences of the first-stage decisions, and depend on the specific realization of the uncertain parameters (the "scenario"). These are usually represented by the variable vector `y_s`, where `s` represents the scenario.
*   **Uncertainty:** The uncertain parameters are typically represented by a random variable with a known (or estimated) probability distribution.  We often discretize the uncertainty into a finite set of scenarios, each with a corresponding probability `p_s`.
*   **Deterministic Equivalent:** The DEF replaces the stochastic program with a deterministic optimization problem that explicitly includes variables for each scenario of the second-stage decisions.  It incorporates constraints and the objective function to account for all scenarios and their probabilities.

**How we can use it:**

The DEF allows us to solve stochastic programs using standard deterministic optimization solvers. By explicitly accounting for all scenarios, it ensures that the first-stage decisions are robust to the uncertainty.

**Mathematical Representation (General Form):**

Consider a two-stage stochastic linear program with the following general form:

Minimize:  `c'x + E_ξ[Q(x, ξ)]`
Subject to:
`Ax = b`
`x >= 0`

Where:

*   `x`: First-stage decision variables.
*   `c`: Cost vector for the first-stage variables.
*   `ξ`: Random variable representing the uncertain parameters.
*   `E_ξ[Q(x, ξ)]`: The expected value of the second-stage cost.
*   `A`: First-stage constraint matrix.
*   `b`: First-stage constraint vector.
*   `Q(x, ξ)`: Optimal value of the second-stage problem, given the first-stage decision `x` and the realization of uncertainty `ξ`. The second stage problem is usually of the form:
Minimize: `q'(ξ)y(ξ)`
Subject to:
`T(ξ)x + W(ξ)y(ξ) = h(ξ)`
`y(ξ) >= 0`

Where:
* `y(ξ)`: Second-stage decision variables given scenario `ξ`.
* `q'(ξ)`: Cost vector for the second-stage variables given scenario `ξ`.
* `T(ξ)`: Technology matrix linking first and second stage decisions given scenario `ξ`.
* `W(ξ)`: Recourse matrix given scenario `ξ`.
* `h(ξ)`: Right-hand side vector of the second stage constraint given scenario `ξ`.

The deterministic equivalent of this two-stage problem (assuming `S` scenarios) is:

Minimize: `c'x + ∑_{s=1}^{S} p_s * q_s' y_s`
Subject to:
`Ax = b`
`T_s x + W_s y_s = h_s` for all `s = 1, ..., S`
`x >= 0`
`y_s >= 0` for all `s = 1, ..., S`

Where:

*   `s`: Index for scenario `s`.
*   `p_s`: Probability of scenario `s`.
*   `q_s`: Second-stage cost vector for scenario `s`.
*   `y_s`: Second-stage decision variables for scenario `s`.
*   `T_s`, `W_s`, `h_s`: Scenario-specific matrices and vectors for the second-stage constraints. These are scenario-dependent realizations of the stochastic parameters `T(ξ)`, `W(ξ)`, and `h(ξ)`.
## 2) Application scenario

**Supply Chain Management with Demand Uncertainty:**

A company needs to decide how much inventory to order (first-stage decision, `x`) *before* knowing the actual demand. After the demand is revealed, the company can make adjustments (second-stage decisions, `y_s`) such as purchasing additional inventory at a higher cost, or selling excess inventory at a discounted price. The uncertainty lies in the customer demand.

In this scenario:

*   `x`: Represents the initial order quantity for each product.
*   `y_s`:  Represents the amount of additional inventory purchased (or excess inventory sold) for each product in scenario `s`.
*   `ξ`: Represents the uncertain demand for each product.
*   `c`: Represents the initial ordering cost per unit.
*   `q_s`: Represents the cost of purchasing additional inventory or the revenue from selling excess inventory in scenario `s`.
*   `A` and `b`: might represent budget constraints on the initial order quantity.
*   `T_s`, `W_s`, and `h_s`: Represents the constraints linking the initial order quantity to the demand, additional purchase/sale quantities, and inventory levels in each scenario. `h_s` would depend on the demand realization for that scenario.

The objective is to minimize the total cost, which includes the initial ordering cost *plus* the expected cost of the second-stage decisions (adjustments to the initial order based on the realized demand). The DEF would explicitly include a variable for the additional inventory purchased/sold for each scenario of possible demand. This allows the company to determine an initial order quantity that minimizes the expected total cost across all possible demand scenarios.
## 3) Python method (if possible)

Here's an example using the `pyomo` library to formulate and solve a Deterministic Equivalent for a simplified two-stage stochastic program:

```python
from pyomo.environ import *
import numpy as np

# Define the number of scenarios
num_scenarios = 3

# Scenario probabilities
scenario_probs = [0.3, 0.4, 0.3]

# Scenario-specific data (example: right-hand side of second-stage constraints)
h_values = [5, 7, 9]

# Create a concrete Pyomo model
model = ConcreteModel()

# First-stage variable
model.x = Var(within=NonNegativeReals)

# Second-stage variables (indexed by scenario)
model.y = Var(range(num_scenarios), within=NonNegativeReals)

# Objective function
model.objective = Objective(
    expr=model.x + sum(scenario_probs[s] * model.y[s] for s in range(num_scenarios)),
    sense=minimize,
)

# First-stage constraint
model.constraint1 = Constraint(expr=model.x >= 2)

# Second-stage constraints (indexed by scenario)
def second_stage_constraint_rule(model, s):
    #Simplified to x + y[s] >= h_values[s]
    return model.x + model.y[s] >= h_values[s]

model.second_stage_constraints = Constraint(range(num_scenarios), rule=second_stage_constraint_rule)


# Solve the model
solver = SolverFactory('glpk')  # Or use another solver like 'ipopt'
solver.solve(model)

# Print the results
print("First-stage decision (x):", model.x.value)
for s in range(num_scenarios):
    print(f"Second-stage decision (y_{s}):", model.y[s].value)
print("Objective Value:", model.objective.expr())
```

**Explanation:**

1.  **Import Libraries:** Imports `pyomo.environ` for modeling and `numpy` for numerical operations.
2.  **Define Scenarios and Data:** Defines the number of scenarios, their probabilities, and scenario-specific data (e.g., `h_values`).
3.  **Create Pyomo Model:** Creates a concrete Pyomo model instance.
4.  **Define Variables:** Defines the first-stage variable `x` and the second-stage variables `y[s]` indexed by scenario.
5.  **Define Objective Function:** Defines the objective function, which is the sum of the first-stage cost and the expected second-stage cost (weighted by scenario probabilities).
6.  **Define Constraints:** Defines the first-stage constraint and the second-stage constraints, indexed by scenario.  This example simplifies the second-stage constraint structure for brevity.  The general form would involve matrices `T_s`, `W_s`, and vectors `h_s`.
7.  **Solve the Model:** Creates a solver instance (using `glpk` in this case), solves the model, and prints the results.

**Important Notes:**

*   This is a *very* simplified example. Real-world stochastic programs can have much more complex objective functions and constraints.
*   The choice of solver is important. `glpk` is suitable for small to medium-sized *linear* problems. For larger or non-linear problems, consider solvers like `ipopt`, `cplex`, or `gurobi`.
*   Scenario generation is a critical aspect of stochastic programming.  The quality of the solution depends heavily on the representativeness of the chosen scenarios. Techniques like Monte Carlo simulation, scenario trees, and moment matching are used to generate scenarios.
* The technology matrix `T_s` and the recourse matrix `W_s` will dictate how the first stage decisions influence the second stage decisions based on the realized scenario.

## 4) Follow-up question

How does the size of the Deterministic Equivalent Formulation grow as the number of scenarios increases? What are some techniques for dealing with a large number of scenarios to make the problem computationally tractable?
---
title: "Introduction to Stochastic Programming (Why uncertainty changes optimization)"
date: "2026-02-11"
week: 7
lesson: 2
slug: "introduction-to-stochastic-programming-why-uncertainty-changes-optimization"
---

# Topic: Introduction to Stochastic Programming (Why uncertainty changes optimization)

## 1) Formal definition (what is it, and how can we use it?)

Stochastic Programming (SP) is a framework for modeling optimization problems that involve uncertainty. Unlike deterministic optimization, where all parameters are known with certainty, SP explicitly incorporates random variables representing uncertain parameters. This is crucial when decisions must be made before the actual values of these parameters are revealed.

**What is it?**

Formally, an SP problem can be represented as follows:

`min f(x, ξ)`
subject to
`g(x, ξ) ≤ 0`
`x ∈ X`

where:

*   `x` is the vector of decision variables (the variables we can control).
*   `ξ` is a random vector (representing uncertain parameters) with a known probability distribution. This distribution can be discrete (e.g., a set of scenarios with associated probabilities) or continuous.
*   `f(x, ξ)` is the objective function, which depends on both the decision variables and the random parameters.
*   `g(x, ξ) ≤ 0` represents the constraints, which may also depend on both the decision variables and the random parameters.
*   `X` is the feasible region for the decision variables.

The key difference from deterministic optimization lies in the presence of `ξ`.  Because `ξ` is uncertain, we can't directly minimize `f(x, ξ)` or ensure `g(x, ξ) ≤ 0` for all possible values of `ξ`. Instead, we optimize with respect to the *expected* value or some other measure of the objective function and ensure constraints are satisfied in a probabilistic sense.

Common approaches within Stochastic Programming include:

*   **Two-Stage Stochastic Programming:** Decisions are made in two stages. In the first stage, *here-and-now* decisions `x` are made before the realization of the uncertainty `ξ`. In the second stage, after `ξ` is observed, *recourse* actions `y(x, ξ)` are taken to mitigate the impact of the uncertainty.  The goal is to minimize the expected cost over all possible realizations of `ξ`, considering both the first-stage cost and the expected second-stage (recourse) cost.
*   **Chance-Constrained Programming:** Some constraints must hold only with a specified probability (e.g., `P(g(x, ξ) ≤ 0) ≥ α`, where α is a probability level, such as 0.95). This is useful when strict constraint satisfaction is not always necessary or feasible.
*   **Multi-Stage Stochastic Programming:** Decisions are made sequentially over multiple time periods, with uncertainty revealed at each stage. This leads to a more complex, dynamic decision-making process.

**How can we use it?**

SP is used to make robust decisions in the face of uncertainty. It helps to:

*   Quantify the impact of uncertainty on the optimal solution.
*   Find solutions that are less sensitive to variations in uncertain parameters.
*   Balance the trade-off between expected performance and risk.
*   Make decisions that are feasible under a range of possible scenarios.
*   Find optimal recourse actions to react to realized uncertainty.

## 2) Application scenario

**Example: Supply Chain Optimization with Uncertain Demand**

A company needs to determine the optimal inventory levels for its products. The demand for these products is uncertain and can vary significantly depending on market conditions.

**Deterministic Approach:** The company might use historical data to estimate the average demand and then use this average value in a deterministic optimization model to determine the optimal inventory levels. However, this approach ignores the variability in demand.  If the actual demand is much higher than the average, the company will experience stockouts and lost sales. If the actual demand is much lower, the company will incur excessive inventory holding costs.

**Stochastic Programming Approach:**

The company can use stochastic programming to explicitly model the uncertainty in demand. Let:

*   `x` be the vector of inventory levels for each product (decision variables).
*   `ξ` be a random vector representing the demand for each product.  We might represent `ξ` as a set of scenarios, each with a different demand level and an associated probability. For example:
    *   Scenario 1: Demand is high (probability 0.3)
    *   Scenario 2: Demand is medium (probability 0.5)
    *   Scenario 3: Demand is low (probability 0.2)
*   `f(x, ξ)` be the cost function, including inventory holding costs, shortage costs, and ordering costs.
*   `g(x, ξ)` be constraints, such as warehouse capacity constraints or minimum service level constraints.

The stochastic programming problem could be formulated as a two-stage problem. The first stage decision `x` determines the inventory levels *before* knowing the actual demand `ξ`. The second stage recourse decision involves ordering additional units or offering discounts to clear excess inventory *after* the demand `ξ` is realized. The objective is to minimize the expected total cost (first-stage inventory cost plus expected second-stage recourse cost).

By solving this stochastic programming problem, the company can find inventory levels that are robust to the uncertainty in demand and that minimize the expected total cost, balancing the risks of stockouts and excessive inventory.

## 3) Python method (if possible)

While a complete implementation of a stochastic program depends on the problem structure and the chosen solver, we can illustrate a simple two-stage stochastic program using the `pyomo` library.

```python
from pyomo.environ import *
import numpy as np

# Define the model
model = ConcreteModel()

# Define scenarios (for simplicity, just 2 scenarios)
scenarios = ['high', 'low']
probabilities = {'high': 0.6, 'low': 0.4}

# First-stage decision variable (e.g., quantity to order)
model.x = Var(domain=NonNegativeReals, name="OrderQuantity")

# Second-stage decision variables (recourse actions)
model.y_high = Var(domain=NonNegativeReals, name="SalesHigh") # Sales in high demand scenario
model.y_low = Var(domain=NonNegativeReals, name="SalesLow") # Sales in low demand scenario
model.z_high = Var(domain=NonNegativeReals, name="ShortageHigh") # Shortage in high demand scenario
model.z_low = Var(domain=NonNegativeReals, name="ShortageLow") # Shortage in low demand scenario

# Scenario-dependent parameters (demands)
demand = {'high': 100, 'low': 50}

# Objective function (minimize cost)
ordering_cost = 1.0  # Cost per unit ordered
shortage_cost = 5.0 # Cost per unit short
excess_cost = 0.5   # Cost per excess unit

def objective_rule(model):
    first_stage_cost = ordering_cost * model.x
    expected_second_stage_cost = (
        probabilities['high'] * (shortage_cost * model.z_high + excess_cost * (model.x - model.y_high)) +
        probabilities['low'] * (shortage_cost * model.z_low + excess_cost * (model.x - model.y_low))
    )
    return first_stage_cost + expected_second_stage_cost

model.objective = Objective(rule=objective_rule, sense=minimize)

# Constraints for high demand scenario
def sales_high_constraint_rule(model):
    return model.y_high + model.z_high == demand['high']
model.sales_high_constraint = Constraint(rule=sales_high_constraint_rule)

def sales_high_limit_rule(model):
    return model.y_high <= model.x
model.sales_high_limit = Constraint(rule=sales_high_limit_rule)

# Constraints for low demand scenario
def sales_low_constraint_rule(model):
    return model.y_low + model.z_low == demand['low']
model.sales_low_constraint = Constraint(rule=sales_low_constraint_rule)

def sales_low_limit_rule(model):
    return model.y_low <= model.x
model.sales_low_limit = Constraint(rule=sales_low_limit_rule)


# Solve the model
solver = SolverFactory('glpk')  # Or another solver like 'ipopt'
solver.solve(model)

# Print the results
print("Optimal Order Quantity:", value(model.x))
print("Expected Cost:", value(model.objective))
print("High Demand Sales:", value(model.y_high))
print("Low Demand Sales:", value(model.y_low))
print("High Demand Shortage:", value(model.z_high))
print("Low Demand Shortage:", value(model.z_low))
```

**Explanation:**

*   We define a simple two-stage problem with two scenarios ("high" and "low" demand).
*   `model.x` is the first-stage decision variable (the quantity to order before observing the demand).
*   `model.y_high`, `model.y_low`, `model.z_high`, `model.z_low` are second-stage variables.  `y` represents the number of sales and `z` represents the number of shortages.
*   The objective function minimizes the expected cost, including ordering cost, shortage cost, and excess inventory cost, weighted by the scenario probabilities.
*   The constraints ensure that the total demand is met (sales + shortage = demand) and that sales do not exceed the ordered quantity.
*   The code then solves the model using a solver (`glpk` is a simple open-source solver), and prints the optimal solution.

This is a simplified example; real-world stochastic programs often involve many more scenarios, variables, and constraints. Libraries like `pyomo` and `gurobipy` (for Gurobi users) provide powerful tools for modeling and solving these complex problems.  The *scenariogen* extension for pyomo is also useful for generating scenarios.

## 4) Follow-up question

How do the computational complexity and scalability challenges of stochastic programming affect the choice of solution methods and approximation techniques, especially when dealing with a large number of scenarios or continuous random variables?  Specifically, what are some commonly used decomposition or sampling techniques for solving large-scale stochastic programs?
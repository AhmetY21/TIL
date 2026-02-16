---
title: "Mean-CVaR Stochastic Programming (two-stage)"
date: "2026-02-16"
week: 8
lesson: 2
slug: "mean-cvar-stochastic-programming-two-stage"
---

# Topic: Mean-CVaR Stochastic Programming (two-stage)

## 1) Formal definition (what is it, and how can we use it?)

Mean-CVaR Stochastic Programming is a class of stochastic programming problems that aims to optimize a decision based on both the expected value (mean) and the Conditional Value-at-Risk (CVaR) of the resulting cost/profit, specifically in a two-stage setting.  The goal is to balance the desire for high expected return with the need to mitigate potential downside risk represented by CVaR.

**What is it?**

*   **Stochastic Programming:** Optimization problems where some of the parameters are uncertain and represented by random variables.
*   **Two-Stage:** Decisions are made in two stages.  First-stage (here-and-now) decisions are made before the uncertainty is revealed.  Second-stage (recourse) decisions are made after the uncertainty is realized, to compensate for the consequences of the first-stage decisions.
*   **Mean-CVaR:** The objective function combines the expected value of the outcome with the CVaR of the outcome.  A common formulation involves minimizing a weighted sum of the negative expected value (to maximize the expected outcome) and the CVaR.  This allows the decision-maker to control the trade-off between expected return and risk.
*   **CVaR (Conditional Value-at-Risk):** Also known as Expected Shortfall, CVaR is a risk measure that quantifies the expected loss given that the loss exceeds a certain threshold (Value-at-Risk or VaR).  It's a more coherent risk measure than VaR because it considers the severity of losses beyond the VaR level.

**How can we use it?**

Mean-CVaR stochastic programming allows us to:

*   **Make robust decisions under uncertainty:** By considering a range of possible scenarios and their probabilities, we can find solutions that perform well across different circumstances.
*   **Control risk:**  The CVaR term explicitly manages the potential for large losses. The weight assigned to CVaR in the objective function determines the level of risk aversion. Higher weights emphasize risk reduction.
*   **Handle recourse actions:** The two-stage structure allows for incorporating corrective actions taken after the uncertain parameters are revealed.  This makes the model more realistic and allows for flexibility in response to different scenarios.
*   **Model real-world problems:** Many problems naturally fit the two-stage stochastic programming framework, such as supply chain management, energy planning, and financial portfolio optimization.

The general mathematical formulation looks like this (minimization problem):

```
min  c'x + α * E[Q(x, ξ)] + (1 - α) * CVaR_β(Q(x, ξ))

subject to:
Ax = b
x >= 0
```

Where:

*   `x` is the vector of first-stage decision variables.
*   `c` is the cost vector associated with the first-stage decisions.
*   `Q(x, ξ)` is the optimal value of the second-stage problem, which depends on the first-stage decision `x` and the realization of the uncertain parameter `ξ`.  This represents the recourse cost.
*   `ξ` is a random vector representing the uncertain parameters.
*   `α` is the weight for the expected value.
*   `(1 - α)` is the weight for the CVaR.
*   `CVaR_β(Q(x, ξ))` is the CVaR of the second-stage cost at confidence level `β`. Common values for beta are 0.9 or 0.95.
*   `A` and `b` define constraints on the first-stage decision variables.

The second-stage problem, for a given scenario `ξ` and first-stage decision `x`, typically looks like this (minimization):

```
Q(x, ξ) = min q(ξ)'y
subject to:
T(ξ)x + W(ξ)y = h(ξ)
y >= 0
```

Where:

*   `y` is the vector of second-stage decision variables.
*   `q(ξ)` is the cost vector associated with the second-stage decisions, dependent on the scenario.
*   `T(ξ)`, `W(ξ)`, and `h(ξ)` define the constraints on the second-stage decisions, dependent on the scenario and first-stage decisions.

## 2) Application scenario

**Example: Supply Chain Network Design under Demand Uncertainty**

A company needs to design a supply chain network by deciding on the capacity of warehouses to build in different locations (first-stage decision). The demand for the company's products in each region is uncertain (random variable). After the demand is realized, the company can decide how much product to ship from each warehouse to each region to satisfy demand (second-stage decision).

**Decision Variables:**

*   `x_i`: Capacity of warehouse built in location `i` (first-stage).
*   `y_{ij}(ω)`: Amount of product shipped from warehouse `i` to region `j` in scenario `ω` (second-stage).

**Objective:**

Minimize the total cost, considering both building the warehouses and shipping the products. The objective function includes:

*   Cost of building warehouses based on their capacity.
*   Expected cost of shipping products, considering all possible demand scenarios.
*   CVaR of the shipping costs, to protect against scenarios with very high shipping costs (e.g., unexpectedly high demand).

**Constraints:**

*   Warehouse capacity constraints (first-stage).
*   Demand satisfaction constraints: Total amount shipped to each region must meet the demand in that region in each scenario (second-stage).
*   Warehouse capacity limits: Amount shipped from each warehouse cannot exceed its capacity in each scenario (second-stage).

**Uncertain Parameter:**

*   `d_j(ω)`: Demand in region `j` in scenario `ω`.

**Benefits of using Mean-CVaR:**

The Mean-CVaR approach allows the company to balance the cost of building larger warehouses (to meet potential high demand) with the risk of building too much capacity (if demand turns out to be low). By controlling the weight on the CVaR term, the company can specify how risk-averse it wants to be.

## 3) Python method (if possible)

While there isn't a single function specifically for *Mean-CVaR two-stage stochastic programming*, we can implement it using general-purpose optimization libraries like Pyomo or Gurobi with its Python API. Here's a conceptual outline using Pyomo, illustrating how to model this:

```python
from pyomo.environ import *
import numpy as np

# Define the model
model = ConcreteModel()

# -------------------- First-Stage Variables --------------------
# Example: Warehouse capacity
num_warehouses = 2
model.x = Var(range(num_warehouses), domain=NonNegativeReals)  # Warehouse capacities

# -------------------- Scenario Generation (Simplified)--------------------
# For illustration, let's say we have 3 scenarios for demand
num_scenarios = 3
scenario_probabilities = [0.3, 0.4, 0.3]  # Probabilities of each scenario
demands = {
    0: [10, 15],  # Scenario 0: Demand in region 1 and region 2
    1: [12, 18],  # Scenario 1: Demand in region 1 and region 2
    2: [8, 12]   # Scenario 2: Demand in region 1 and region 2
}

# -------------------- Second-Stage Variables (Scenario-Dependent) --------------------
# Example: Shipping amounts from warehouses to regions
num_regions = 2
model.y = Var(range(num_scenarios), range(num_warehouses), range(num_regions), domain=NonNegativeReals) #shipping from warehouse i to region j, scenario omega

# -------------------- CVaR Variables --------------------
alpha = 0.95  # Confidence level for CVaR
cvar_weight = 0.5 # weight on the CVaR (1-alpha from section 1)
expected_value_weight = 1-cvar_weight # weight on the mean (alpha from section 1)
model.z = Var(domain=Reals) # VaR variable
model.v = Var(range(num_scenarios), domain=NonNegativeReals) # Exceedance variable


# -------------------- Objective Function --------------------
def objective_rule(model):
    # First-stage cost (warehouse building cost)
    first_stage_cost = sum(model.x[i] for i in range(num_warehouses)) # Simple example: cost = capacity
    # Second-stage cost (expected shipping cost)
    expected_second_stage_cost = sum(scenario_probabilities[s] * sum(model.y[s,i,j] for i in range(num_warehouses) for j in range(num_regions)) for s in range(num_scenarios)) #shipping cost is the amount shipped

    # CVaR term
    cvar_term = model.z + (1/(1-alpha)) * sum(scenario_probabilities[s] * model.v[s] for s in range(num_scenarios))

    return first_stage_cost + expected_value_weight * expected_second_stage_cost + cvar_weight * cvar_term #combine objective components

model.objective = Objective(rule=objective_rule, sense=minimize)

# -------------------- Constraints --------------------
# Example: Demand satisfaction constraints (scenario-dependent)
def demand_satisfaction_rule(model, s, j):
    return sum(model.y[s,i,j] for i in range(num_warehouses)) >= demands[s][j]
model.demand_satisfaction = Constraint(range(num_scenarios), range(num_regions), rule=demand_satisfaction_rule)

# Example: Warehouse capacity constraints (scenario-dependent)
def capacity_constraint_rule(model, s, i):
    return sum(model.y[s,i,j] for j in range(num_regions)) <= model.x[i]
model.capacity_constraint = Constraint(range(num_scenarios), range(num_warehouses), rule=capacity_constraint_rule)

# CVaR constraints
def cvar_def_rule(model, s):
  second_stage_cost_scenario = sum(model.y[s,i,j] for i in range(num_warehouses) for j in range(num_regions)) #shipping cost is the amount shipped
  return model.z + model.v[s] >= second_stage_cost_scenario
model.cvar_def = Constraint(range(num_scenarios), rule=cvar_def_rule)

# v >= 0, exceedence variable constraint is already defined


# -------------------- Solve the model --------------------
solver = SolverFactory('glpk') # or 'gurobi', 'cplex'
solver.solve(model)

# -------------------- Print results --------------------
print("Warehouse capacities:")
for i in range(num_warehouses):
    print(f"Warehouse {i}: {model.x[i].value}")

print("\nShipping amounts:")
for s in range(num_scenarios):
    print(f"Scenario {s}:")
    for i in range(num_warehouses):
        for j in range(num_regions):
            print(f"  From warehouse {i} to region {j}: {model.y[s,i,j].value}")

print("\nVaR:")
print(model.z.value)

print("\nExceedence variables:")
for s in range(num_scenarios):
    print(model.v[s].value)
```

**Explanation:**

1.  **Model Definition:** We use Pyomo's `ConcreteModel` to define the optimization model.
2.  **First-Stage Variables:** `model.x` represents the first-stage decision variables (warehouse capacities).
3.  **Scenario Generation:** We create a simplified scenario set for the uncertain demand. In a real application, you might use historical data or simulation to generate scenarios.
4.  **Second-Stage Variables:** `model.y` represents the second-stage decision variables (shipping amounts), which depend on the scenario.
5.  **CVaR Variables:** `model.z` is the VaR variable (Value at Risk), and `model.v` is the exceedance variable that models (Q(x, ξ) - z)^+. This is a linear representation of CVaR.
6.  **Objective Function:** The objective function minimizes the total cost, including the first-stage cost (warehouse building cost), the expected second-stage cost (shipping cost), and the CVaR of the shipping cost.  The `alpha` parameter (here named `cvar_weight` and `expected_value_weight`) controls the trade-off between expected cost and risk.
7.  **Constraints:** The constraints enforce demand satisfaction (shipping amounts must meet demand in each scenario), warehouse capacity limits (shipping amounts cannot exceed warehouse capacity), and CVaR formulation.
8.  **Solver:** We use the `glpk` solver (you can use `gurobi` or `cplex` if you have them installed).
9.  **Results:** The code prints the optimal warehouse capacities and shipping amounts for each scenario.

**Important Notes:**

*   This is a simplified example. A real-world application would involve more complex constraints and a more sophisticated scenario generation process.
*   Using a commercial solver like Gurobi or CPLEX is generally recommended for solving large-scale stochastic programming problems.
*   The CVaR formulation used here is a common linear representation that makes the problem solvable using linear programming solvers.
*   This example assumes discrete scenarios. For continuous distributions, you'll need to use approximation techniques, such as scenario sampling or decomposition methods.

## 4) Follow-up question

How does the choice of the confidence level (β) for CVaR and the weight assigned to the CVaR term (1 - α) in the objective function affect the optimal solution, and what are some guidelines for choosing appropriate values for these parameters in a real-world application?
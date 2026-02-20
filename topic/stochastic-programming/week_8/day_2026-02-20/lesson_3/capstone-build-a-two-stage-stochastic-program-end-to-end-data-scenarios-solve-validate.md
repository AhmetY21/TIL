---
title: "Capstone: Build a Two-Stage Stochastic Program End-to-End (data → scenarios → solve → validate)"
date: "2026-02-20"
week: 8
lesson: 3
slug: "capstone-build-a-two-stage-stochastic-program-end-to-end-data-scenarios-solve-validate"
---

# Topic: Capstone: Build a Two-Stage Stochastic Program End-to-End (data → scenarios → solve → validate)

## 1) Formal definition (what is it, and how can we use it?)

A two-stage stochastic program is a mathematical optimization model used for decision-making under uncertainty. It involves making decisions in two stages:

*   **First-stage (Here-and-Now) Decisions:** These decisions must be made *before* the uncertainty is revealed. They are independent of the realization of the random variables. These decisions define an initial plan that can be adjusted.

*   **Second-stage (Recourse) Decisions:** These decisions are made *after* the uncertainty is revealed. They are adjustments to the first-stage decisions and aim to minimize the cost of deviations from the initial plan due to the specific outcome of the random variables. These decisions are scenario-dependent. These decisions define the adaptation strategy.

**Formally:**

A general two-stage stochastic linear program can be represented as follows:

```
min cᵀx + E[Q(x, ω)]
subject to:
Ax = b
x ≥ 0
```

where:

*   `x` is the vector of first-stage decision variables.
*   `c` is the cost vector associated with the first-stage decisions.
*   `ω` is the random variable representing the uncertainty.  It can take various realizations (scenarios).
*   `E[Q(x, ω)]` is the expected value of the second-stage cost function, which depends on the first-stage decision `x` and the realization of the random variable `ω`.  This represents the average cost across all possible scenarios.
*   `A` and `b` are constraint matrices and vectors for the first-stage constraints.

The second-stage cost function `Q(x, ω)` is defined as:

```
Q(x, ω) = min q(ω)ᵀy(ω)
subject to:
T(ω)x + W(ω)y(ω) = h(ω)
y(ω) ≥ 0
```

where:

*   `y(ω)` is the vector of second-stage decision variables for scenario `ω`.
*   `q(ω)` is the cost vector associated with the second-stage decisions for scenario `ω`.
*   `T(ω)`, `W(ω)`, and `h(ω)` are constraint matrices and vectors for the second-stage constraints, which depend on the scenario `ω`. `T(ω)` models the impact of the first-stage decisions on the second-stage constraints. `W(ω)` couples the recourse actions with the uncertain parameters `ω`.

**How to use it:**

Two-stage stochastic programming is used when you need to make decisions now, but the outcome depends on future events you don't know for sure. By considering a range of possible scenarios and optimizing for the expected outcome, it allows for more robust and resilient decisions compared to deterministic optimization. The "end-to-end" approach involves the entire process, covering data collection to validation:

1.  **Data:** Gathering historical data or expert opinions to characterize the uncertainty.
2.  **Scenarios:** Generating a set of representative scenarios based on the data. This can involve statistical methods, simulation, or expert judgement. Each scenario represents a possible future outcome.
3.  **Solve:** Formulating the two-stage stochastic program and solving it using an optimization solver. This yields the optimal first-stage decisions and the optimal second-stage decisions for each scenario.
4.  **Validate:** Evaluating the solution's performance under various conditions, including scenarios not used in the initial optimization. This ensures the solution is robust and reliable. This often involves simulation or out-of-sample testing.

## 2) Application scenario

**Supply Chain Management with Demand Uncertainty:**

A company needs to decide how much inventory to stock in a warehouse before the start of the sales season (first-stage decision). The demand for the product during the season is uncertain (random variable). After the season starts, the company observes the actual demand. If demand is higher than the initial inventory, the company can order more products at a higher cost (second-stage decision - recourse action). If demand is lower than the initial inventory, the company can sell the excess inventory at a discounted price.

*   **First-stage decision (x):** Amount of inventory to stock.
*   **Random variable (ω):** Demand during the season.
*   **Scenarios:**  Possible demand levels (e.g., low, medium, high).
*   **Second-stage decision (y(ω)):** Amount of additional products to order (if demand is high) or amount of excess inventory to sell at a discount (if demand is low).
*   **Objective:** Minimize the total cost, including the cost of initial inventory, the cost of additional orders (if needed), and the revenue from selling excess inventory.

The company would gather historical demand data, generate demand scenarios, formulate the two-stage stochastic program, solve it to determine the optimal initial inventory level, and then validate the solution by simulating the supply chain's performance under different demand scenarios.  The validation process would help determine if the chosen number of scenarios and their probabilities accurately reflect the risks and opportunities of demand variability.

## 3) Python method (if possible)

We can use the `Pyomo` library in Python to model and solve two-stage stochastic programs.  Since a complete end-to-end implementation (data collection, scenario generation, etc.) is extensive, I will provide a simplified example demonstrating the core Pyomo code for defining and solving the model, assuming scenarios are already generated.

```python
from pyomo.environ import *
import numpy as np

# Define the number of scenarios
num_scenarios = 3

# Define scenario probabilities (must sum to 1)
scenario_probabilities = [0.3, 0.4, 0.3]

# Define scenario data (example demand values)
scenario_demands = [80, 100, 120]

# Create a concrete Pyomo model
model = ConcreteModel()

# First-stage variable: Inventory level
model.x = Var(domain=NonNegativeReals)

# Scenario set
model.Scenarios = RangeSet(1, num_scenarios)

# Second-stage variables (indexed by scenario):
# y[s]: Amount to order in scenario s
model.y = Var(model.Scenarios, domain=NonNegativeReals)
# z[s]: Amount of excess inventory sold at discount in scenario s
model.z = Var(model.Scenarios, domain=NonNegativeReals)


# Objective function: Minimize total cost
# c1: cost per unit of initial inventory
# c2: cost per unit of additional order
# c3: revenue per unit of excess inventory sold
c1 = 5
c2 = 7
c3 = 3

def objective_rule(model):
  return c1 * model.x + sum(scenario_probabilities[s-1] * (c2 * model.y[s] - c3 * model.z[s]) for s in model.Scenarios)

model.objective = Objective(rule=objective_rule, sense=minimize)

# Constraint: Demand must be met in each scenario
def demand_rule(model, s):
  return model.x + model.y[s] - model.z[s] >= scenario_demands[s-1]

model.demand_constraint = Constraint(model.Scenarios, rule=demand_rule)


# Solve the model
solver = SolverFactory('glpk')  # You can use other solvers like 'gurobi', 'cplex', etc.
solver.solve(model)


# Print the results
print("First-stage decision (Inventory Level):", model.x.value)
for s in model.Scenarios:
  print(f"Scenario {s}:")
  print(f"  Additional order: {model.y[s].value}")
  print(f"  Excess inventory sold: {model.z[s].value}")
```

**Explanation:**

1.  **Scenario Definition:** The code defines a set of scenarios and their probabilities. The `scenario_demands` variable holds the demand values for each scenario.
2.  **Pyomo Model:** A `ConcreteModel` is created to represent the optimization problem.
3.  **Variables:** The `x` variable represents the first-stage inventory level. The `y` and `z` variables represent the second-stage decisions (amount to order and amount to sell at a discount, respectively), indexed by scenario.
4.  **Objective Function:** The objective function minimizes the expected total cost, considering the cost of the initial inventory, the cost of additional orders, and the revenue from selling excess inventory, weighted by the scenario probabilities.
5.  **Constraints:** The `demand_constraint` ensures that the demand is met in each scenario, either through the initial inventory or additional orders.
6.  **Solver:**  The code uses the GLPK solver to find the optimal solution. Other solvers like Gurobi or CPLEX can be used for larger and more complex problems.
7.  **Results:**  The code prints the optimal first-stage decision (inventory level) and the optimal second-stage decisions (additional order and excess inventory sold) for each scenario.

**Important Notes:**

*   This is a simplified example.  Real-world problems often have more complex constraints and objective functions.
*   Scenario generation is a crucial step in stochastic programming. This example assumes scenarios are already provided. Various methods exist for scenario generation, including historical data analysis, Monte Carlo simulation, and scenario reduction techniques.
*   Solver selection is important.  Gurobi and CPLEX are commercial solvers that are often preferred for large-scale optimization problems, but GLPK is a free open-source alternative suitable for smaller problems.
*   Model validation is essential to ensure the solution is robust and reliable.

## 4) Follow-up question

How can scenario reduction techniques be used to efficiently solve large-scale two-stage stochastic programs when dealing with a high number of possible scenarios, and what are the trade-offs between accuracy and computational complexity when applying these techniques?
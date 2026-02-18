---
title: "Modeling Stochastic Programs in Pyomo: extensive form (two-stage)"
date: "2026-02-18"
week: 8
lesson: 5
slug: "modeling-stochastic-programs-in-pyomo-extensive-form-two-stage"
---

# Topic: Modeling Stochastic Programs in Pyomo: extensive form (two-stage)

## 1) Formal definition (what is it, and how can we use it?)

The extensive form is a method for solving two-stage stochastic programs by explicitly representing all possible scenarios and formulating a deterministic equivalent of the original stochastic problem.

**What is it?**

A two-stage stochastic program involves making decisions in two stages:

*   **First-stage decisions (Here-and-Now decisions):** These decisions must be made *before* the realization of any uncertain parameters (e.g., demand, costs).
*   **Second-stage decisions (Wait-and-See decisions):** These decisions are made *after* observing the realization of the uncertain parameters. They serve to mitigate the consequences of the first-stage decisions given the observed scenario.

The extensive form transforms the stochastic program into a large, deterministic optimization problem.  For each possible scenario, it defines separate decision variables for the second stage.  The objective function minimizes the expected value of the total cost (first-stage cost plus the average second-stage cost over all scenarios).  Constraints ensure feasibility in both the first and second stages, considering each scenario individually.

**How can we use it?**

The extensive form is useful when:

*   The number of scenarios is relatively small.
*   We need to obtain the optimal first-stage decisions, knowing that we will react optimally in the second stage to any possible outcome.
*   We have a good understanding of the probability distribution of the uncertain parameters.

The extensive form provides the exact optimal solution to the stochastic program (given the specified scenarios and probabilities). However, it can become computationally intractable for problems with a large number of scenarios due to the rapid increase in the number of variables and constraints.

**Mathematical Formulation:**

Let:

*   `x`: First-stage decision variables
*   `y_s`: Second-stage decision variables for scenario `s`
*   `ξ_s`: Realization of uncertain parameters in scenario `s`
*   `p_s`: Probability of scenario `s`
*   `Q`: Set of all scenarios
*   `c`: Cost vector for first-stage decisions
*   `q_s`: Cost vector for second-stage decisions in scenario `s`
*   `T`: Matrix relating first-stage decisions to second-stage constraints
*   `W`: Matrix of second-stage technology coefficients
*   `h_s`: Right-hand side of second-stage constraints in scenario `s`
*   `A`: Matrix of first-stage technology coefficients
*   `b`: Right-hand side of first-stage constraints

The extensive form can be formulated as:

```
min  c'x + Σ_{s ∈ Q} p_s * q_s'y_s
subject to:
   Ax <= b                              (First-stage constraints)
   Tx + Wy_s <= h_s  for all s ∈ Q       (Second-stage constraints, scenario-specific)
   x >= 0
   y_s >= 0  for all s ∈ Q
```

## 2) Application scenario

**Inventory Management with Demand Uncertainty**

A retailer needs to decide how many units of a product to order *now* (first-stage decision) before knowing the actual demand.  The demand is uncertain and can take on a few discrete values (scenarios).  *After* observing the actual demand, the retailer can decide how much to sell or discard excess inventory to meet the demand (second-stage decision).

*   **First-stage decision (x):**  Number of units to order.
*   **Second-stage decisions (y_s):** Number of units to sell and Number of units to discard in scenario `s`.
*   **Uncertain parameters (ξ_s):** Demand in scenario `s`.
*   **Objective:** Minimize the expected total cost, including ordering cost, holding cost for unsold items, and lost sales penalty if demand exceeds the available inventory.
*   **Constraints:** Inventory balance constraints for each scenario, non-negativity constraints.

In this scenario, the extensive form allows the retailer to determine the optimal order quantity (first-stage) by considering all possible demand scenarios and the optimal second-stage responses (selling or discarding) for each scenario, weighted by their probabilities.  A simpler but less accurate approach might use an expected value demand without explicitly optimizing across possible scenario outcomes.

## 3) Python method (if possible)

```python
from pyomo.environ import *

# Define the model
model = ConcreteModel()

# --- Problem Data ---
# Number of scenarios
num_scenarios = 3

# Scenario probabilities
model.p = Param(range(num_scenarios), initialize={0: 0.3, 1: 0.4, 2: 0.3})

# Demands for each scenario
model.demand = Param(range(num_scenarios), initialize={0: 10, 1: 15, 2: 20})

# Ordering cost
ordering_cost = 5

# Selling price
selling_price = 10

# Discard cost (per unit)
discard_cost = 1

# --- First Stage Variables ---
model.x = Var(domain=NonNegativeReals) # Order quantity

# --- Second Stage Variables ---
model.y_sell = Var(range(num_scenarios), domain=NonNegativeReals) # Quantity sold
model.y_discard = Var(range(num_scenarios), domain=NonNegativeReals) # Quantity discarded

# --- Objective Function ---
def objective_rule(model):
  first_stage_cost = ordering_cost * model.x
  second_stage_cost = sum(model.p[s] * (selling_price * model.y_sell[s] - discard_cost * model.y_discard[s]) for s in range(num_scenarios))
  return first_stage_cost - second_stage_cost # Selling price is revenue. Discounting revenue to be subtracted from cost.

model.objective = Objective(rule=objective_rule, sense=minimize)

# --- Constraints ---
# Demand satisfaction constraints for each scenario
def demand_satisfaction_rule(model, s):
  return model.y_sell[s] <= model.demand[s]

model.demand_satisfaction = Constraint(range(num_scenarios), rule=demand_satisfaction_rule)

# Inventory balance constraints for each scenario
def inventory_balance_rule(model, s):
  return model.x == model.y_sell[s] + model.y_discard[s]

model.inventory_balance = Constraint(range(num_scenarios), rule=inventory_balance_rule)


# --- Solve the model ---
solver = SolverFactory('glpk')  # or any other suitable solver
solver.solve(model)

# --- Print the results ---
print("Optimal Order Quantity:", model.x.value)
for s in range(num_scenarios):
  print(f"Scenario {s}: Sell = {model.y_sell[s].value}, Discard = {model.y_discard[s].value}")
```

**Explanation:**

1.  **Model Definition:**  A `ConcreteModel` is created to represent the optimization problem.
2.  **Problem Data:** The scenario probabilities, demands, and cost parameters are defined as `Param` objects.
3.  **Variables:**
    *   `model.x`: The first-stage decision variable representing the order quantity.
    *   `model.y_sell` and `model.y_discard`: Second-stage decision variables representing the quantity sold and discarded for each scenario.
4.  **Objective Function:** The objective function minimizes the expected total cost, which includes the ordering cost minus selling revenue plus discard cost across all scenarios. The summation of weighted scenario costs is handled explicitly. Note the costs are added, but the revenue is subtracted (equivalently, negative costs can be used for revenues).
5.  **Constraints:**
    *   `demand_satisfaction`: Ensures that the quantity sold in each scenario does not exceed the demand.
    *   `inventory_balance`: Ensures that the order quantity equals the sum of the quantity sold and discarded in each scenario.
6.  **Solving:** The model is solved using the `glpk` solver.
7.  **Results:** The optimal order quantity and the quantities sold and discarded for each scenario are printed.

## 4) Follow-up question

How can the extensive form be adapted to handle more complex two-stage stochastic programming problems, such as those with continuous probability distributions or a very large number of discrete scenarios, and what are the limitations of these adaptations?  What alternative approaches are commonly used when the extensive form becomes intractable?
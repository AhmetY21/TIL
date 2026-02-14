---
title: "Feasibility and Penalty Modeling (slack, feasibility recourse)"
date: "2026-02-14"
week: 7
lesson: 4
slug: "feasibility-and-penalty-modeling-slack-feasibility-recourse"
---

# Topic: Feasibility and Penalty Modeling (slack, feasibility recourse)

## 1) Formal definition (what is it, and how can we use it?)

Feasibility and Penalty Modeling, particularly using slack variables and feasibility recourse, addresses situations in Stochastic Programming where obtaining a feasible solution for *every* realization of the uncertain parameters might be impossible or prohibitively expensive. Instead of insisting on absolute feasibility, we relax constraints and penalize infeasibility using either slack variables or feasibility recourse functions in the objective function.

**Slack Variables:**

*   **What it is:**  We introduce artificial non-negative variables (slack variables) to transform hard constraints into soft constraints. These variables represent the amount by which a constraint can be violated.
*   **How we use it:**  We add the slack variables to the constraints, effectively allowing them to be violated.  Then, we add a penalty term to the objective function that is proportional to the magnitude of the slack variables.  This penalty discourages infeasibility but allows it if the cost of violating the constraint is less than the penalty.  This trade-off allows the model to find solutions even when some constraints cannot be strictly satisfied across all scenarios. Let's say we have a constraint `Ax <= b`.  We can introduce a slack variable `s >= 0` and transform it into `Ax - s <= b`. If `s` is positive, it indicates a violation of the original constraint `Ax <= b`.  In the objective function, we would then add a term like `rho * s`, where `rho` is a penalty coefficient.

**Feasibility Recourse (Penalty Function):**

*   **What it is:** Instead of slack variables, we directly add a penalty function to the objective, which represents the cost of violating the constraints. This penalty function depends on the magnitude of the constraint violation.  It quantifies the cost associated with infeasibility in a specific scenario. The penalty function is defined *after* observing the realization of the uncertain parameters.
*   **How we use it:** The penalty function is usually a function of the constraint violations and can take various forms, such as linear, quadratic, or more complex forms.  A common example is to penalize the sum of the violations. The penalty function is added to the original objective function. The goal is to minimize the original objective plus the expected penalty for constraint violations across all possible scenarios.  For example, if we have a constraint `g(x, \omega) <= 0`, where `\omega` is the uncertain parameter, a penalty function could be `P(x, \omega) = \rho * max(0, g(x, \omega))`, where `\rho` is the penalty coefficient. The stochastic program would then aim to minimize `E[f(x, \omega) + P(x, \omega)]`.

**Key Differences:**

*   Slack variables modify the constraints themselves, turning them into soft constraints.
*   Feasibility recourse leaves the constraints as they are but introduces a penalty *only* in the objective function.
*   The choice depends on the specific problem and how naturally the infeasibility can be expressed (and penalized) within the context of the constraints or the objective.

## 2) Application scenario

**Scenario: Production Planning under Uncertain Demand**

A manufacturing company needs to determine the optimal production plan for a product, but the demand is uncertain. The company has limited production capacity and must satisfy demand, or face penalties.

*   **Problem:** The company wants to minimize production costs while meeting uncertain demand. If the demand exceeds the production capacity, the company incurs a shortage cost due to lost sales and potential customer dissatisfaction.
*   **Solution using Slack Variables:** Let `x` be the production quantity and `d_ω` be the demand in scenario `ω`. The constraint to satisfy demand is `x >= d_ω`. Introducing a slack variable `s_ω >= 0` (shortage) transforms the constraint into `x + s_ω >= d_ω`.  The objective function would then be `min  c*x + rho * E[s_ω]`, where `c` is the unit production cost, `rho` is the penalty for each unit of unmet demand, and `E[s_ω]` is the expected shortage.

*   **Solution using Feasibility Recourse:** The constraint remains `x >= d_ω`. The penalty function would be `P(x, d_ω) = rho * max(0, d_ω - x)`. The objective function becomes `min c*x + E[P(x, d_ω)]`, which minimizes the production cost plus the expected penalty for unmet demand.

In both approaches, the stochastic program aims to find the optimal production level `x` that balances production costs with the cost of potential unmet demand under uncertain scenarios. If `rho` is sufficiently high, the model will prioritize fulfilling demand as much as possible.

## 3) Python method (if possible)

Here's an example using Pyomo with slack variables:

```python
from pyomo.environ import *
import numpy as np

# Define the model
model = ConcreteModel()

# Parameters
model.num_scenarios = Param(initialize=3)  # Number of scenarios
scenarios = range(model.num_scenarios())
model.production_cost = Param(initialize=2)  # Cost per unit of production
model.shortage_penalty = Param(initialize=5)  # Penalty per unit of shortage

# Uncertain demand data (replace with your actual data)
demand_data = {0: 10, 1: 15, 2: 12}

# Variables
model.x = Var(domain=NonNegativeReals)  # Production quantity
model.s = Var(scenarios, domain=NonNegativeReals)  # Shortage in each scenario

# Constraints
def demand_rule(model, s):
    return model.x + model.s[s] >= demand_data[s]
model.demand_constraints = Constraint(scenarios, rule=demand_rule)


# Objective Function (Minimize production cost + expected shortage penalty)
def objective_rule(model):
    return model.production_cost * model.x + sum(model.shortage_penalty * model.s[s] for s in scenarios) / model.num_scenarios

model.objective = Objective(rule=objective_rule, sense=minimize)

# Solve the model
solver = SolverFactory('glpk') # or another solver like 'gurobi', 'cplex'
solver.solve(model)

# Print results
print("Optimal Production Quantity:", value(model.x))
for s in scenarios:
    print(f"Shortage in scenario {s}:", value(model.s[s]))
```

**Explanation:**

*   We define the Pyomo model, parameters (production cost, shortage penalty, number of scenarios), and uncertain demand data.
*   `model.x` represents the production quantity, and `model.s` represents the shortage in each scenario (slack variables).
*   `model.demand_constraints` ensures that production plus shortage is greater than or equal to demand in each scenario.
*   `model.objective` minimizes the production cost plus the *expected* shortage penalty.  Dividing by `model.num_scenarios` calculates the average penalty across all scenarios.
*   A solver is used to find the optimal solution.

**Note:**  For feasibility recourse, you would define the penalty function directly within the objective function, without adding slack variables to the constraints. You'd likely need an `if` statement (or `max` function) within the objective function to calculate the penalty based on whether `x` is less than `demand_data[s]`.  The core difference is that the constraint remains `x >= demand_data[s]` and no slack variables are used; the "softness" comes entirely from the penalty in the objective function.

## 4) Follow-up question

How does the choice of penalty coefficient (e.g., `rho` in the examples above) impact the optimal solution, and what methods can be used to determine an appropriate value for it? What are the trade-offs to consider when choosing between a very high penalty versus a relatively small one?
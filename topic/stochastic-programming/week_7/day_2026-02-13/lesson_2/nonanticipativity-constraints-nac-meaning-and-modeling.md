---
title: "Nonanticipativity Constraints (NAC): meaning and modeling"
date: "2026-02-13"
week: 7
lesson: 2
slug: "nonanticipativity-constraints-nac-meaning-and-modeling"
---

# Topic: Nonanticipativity Constraints (NAC): meaning and modeling

## 1) Formal definition (what is it, and how can we use it?)

Nonanticipativity constraints (NACs) are fundamental in stochastic programming, particularly in multi-stage stochastic programming. They enforce the principle that **decisions made at a given stage cannot depend on information that is not yet available at that stage.** In other words, decisions in the current period must be identical across all scenarios that share the same history up to that period. They are designed to maintain *causality*.

**What is it?**

Mathematically, let's say we have a multi-stage stochastic programming problem with *T* stages and a set of scenarios *Ω*. Let *x<sub>t</sub>(ω)* represent the decision variable at stage *t* under scenario *ω*. The nonanticipativity constraints ensure that for all scenarios *ω, ω' ∈ Ω* that have the same history up to stage *t*, the decision *x<sub>t</sub>(ω)* must be equal to *x<sub>t</sub>(ω')*. The "same history" typically refers to the realization of the random variables up to stage *t*.

More formally, if *ξ<sub>t</sub>(ω)* represents the realization of the random variable at stage *t* under scenario *ω*, and if *ξ<sub>1</sub>(ω) = ξ<sub>1</sub>(ω')*, ..., *ξ<sub>t-1</sub>(ω) = ξ<sub>t-1</sub>(ω')*, then we must have *x<sub>t</sub>(ω) = x<sub>t</sub>(ω')*. This means that the decision *x<sub>t</sub>* can only depend on *ξ<sub>1</sub>, ..., ξ<sub>t-1</sub>*.

**How can we use it?**

NACs are typically incorporated into the optimization model as explicit constraints. For example:

`x_t(ω) = x_t(ω')  for all ω, ω' such that ξ_1(ω)=ξ_1(ω'), ..., ξ_{t-1}(ω)=ξ_{t-1}(ω')`

If there are many scenarios, this could lead to a large number of constraints.  The way the scenarios are generated impacts the number of NACs that are needed. For instance, in a scenario tree, NACs are only needed where the tree branches.  Where the tree is a single "trunk", decisions will be the same anyway.

Using NACs ensures that the solution is implementable in reality. Without NACs, the optimization might prescribe different actions based on future information that is not yet known, rendering the solution impractical.  They ensure that decisions can actually be made at each time stage with the information that is available at that stage.

## 2) Application scenario

Consider a supply chain management problem where a company needs to decide how much inventory to order at each stage of a planning horizon. The demand at each stage is uncertain and revealed sequentially. The company needs to decide on its inventory levels at the beginning of each stage, but cannot base its decisions on future demand realizations.

*   **Stage 1:** The company decides how much to order before observing any customer demand.
*   **Stage 2:** The demand for the first period is realized. Knowing the demand, the company decides how much to order for the second period.
*   **Stage 3:** The demand for the second period is realized. The company decides how much to order for the third period.

In this scenario, the inventory ordering decisions at Stage 2 cannot depend on the demand at Stage 3, and the ordering decisions at Stage 3 cannot depend on future (non-existent) demands. NACs are crucial here. They force the ordering decisions for the same past demand realizations to be the same. For example, if the Stage 1 demand is *d<sub>1</sub>*, then regardless of what *d<sub>2</sub>* and *d<sub>3</sub>* may be, the Stage 2 decision *x<sub>2</sub>* must be the same across all scenarios sharing this demand at *t=1*.

Without NACs, an optimization might recommend an ordering strategy that effectively anticipates future demand, which is impossible in a real-world setting.

## 3) Python method (if possible)

While there isn't a single built-in function to *automatically* generate NACs, they can be implemented using optimization libraries like Pyomo or Gurobi (with Python bindings). Here's an example using Pyomo to illustrate how to model NACs:

```python
from pyomo.environ import *

# Sample scenario tree structure (replace with your actual data)
# Each tuple represents a scenario with demand realizations at each stage
scenarios = [
    (10, 15, 20),  # Scenario 1: Demand of 10, 15, 20 at stages 1, 2, 3
    (10, 15, 25),  # Scenario 2: Demand of 10, 15, 25 at stages 1, 2, 3
    (10, 20, 25),  # Scenario 3: Demand of 10, 20, 25 at stages 1, 2, 3
    (15, 20, 25),  # Scenario 4: Demand of 15, 20, 25 at stages 1, 2, 3
    (15, 20, 30)   # Scenario 5: Demand of 15, 20, 30 at stages 1, 2, 3
]

# Stages
stages = range(1, 4)  # Stages 1, 2, 3

# Create a Pyomo model
model = ConcreteModel()

# Decision variables: inventory level at each stage for each scenario
model.inventory = Var(stages, range(len(scenarios)), domain=NonNegativeReals)

# Demand data (from scenarios)
model.demand = Param(range(len(scenarios)), stages, initialize={(s, t): scenarios[s][t-1] for s in range(len(scenarios)) for t in stages})


# Objective function (simplified - minimize total inventory holding cost)
model.obj = Objective(expr=sum(model.inventory[t, s] for t in stages for s in range(len(scenarios))), sense=minimize)

# Nonanticipativity constraints
def nonanticipativity_rule(model, t, scenario1, scenario2):
    if t > 1:  # No NAC at the first stage (all scenarios have the same history)
        # Check if scenarios share the same history up to stage t-1
        same_history = all(model.demand[scenario1, s] == model.demand[scenario2, s] for s in range(1, t))
        if same_history:
            return model.inventory[t, scenario1] == model.inventory[t, scenario2]
        else:
            return Constraint.Skip  # No NAC needed if histories differ
    else:
        return Constraint.Skip # No NAC at the first stage

model.nonanticipativity = Constraint(stages, range(len(scenarios)), range(len(scenarios)), rule=nonanticipativity_rule)

#Example Balance Constraints
def balance_rule(model, s, t):
    if t == 1:
        return model.inventory[t,s] >= model.demand[s, t]  # initial order >= first period demand
    else:
        return model.inventory[t, s] >= model.demand[s,t] + model.inventory[t-1, s] - model.demand[s, t-1]


model.balance = Constraint(range(len(scenarios)), stages, rule=balance_rule)


# Solve the model
solver = SolverFactory('glpk') # Replace with your preferred solver (e.g., Gurobi, CPLEX)
results = solver.solve(model)

# Print the results (example)
if results.solver.termination_condition == TerminationCondition.optimal:
    for t in stages:
        for s in range(len(scenarios)):
            print(f"Stage {t}, Scenario {s}: Inventory = {model.inventory[t, s].value}")
else:
    print("Solver did not find an optimal solution.")
```

**Explanation:**

*   The `scenarios` list represents a simple scenario tree structure.
*   The `demand` parameter stores the demand realizations for each scenario and stage.
*   The `nonanticipativity_rule` function enforces the NACs. It checks if two scenarios share the same demand history up to stage `t-1`. If they do, it adds a constraint that the inventory levels at stage `t` must be equal. If not, it skips the constraint using `Constraint.Skip` (to avoid unnecessary constraints).
* Example `balance_rule` constraint is added for completeness.
*   The solver then finds the optimal inventory levels, satisfying the NACs.

**Important Notes:**

*   This is a simplified example.  Real-world problems will have more complex scenario structures, objective functions, and other constraints.
*   You'll need to adapt the scenario generation and constraint formulation to fit your specific problem.
*   The number of NACs can grow rapidly with the number of scenarios and stages. Techniques like scenario reduction can help manage the problem size.

## 4) Follow-up question

How does the choice of scenario generation technique (e.g., Monte Carlo simulation, scenario trees, moment matching) affect the number and complexity of the nonanticipativity constraints that need to be implemented in a stochastic programming model? Are there scenario generation techniques that are particularly well-suited for problems with many stages and uncertainties, in terms of simplifying the implementation of NACs?
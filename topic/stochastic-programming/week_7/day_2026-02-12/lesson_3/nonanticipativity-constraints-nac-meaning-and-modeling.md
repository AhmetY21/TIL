---
title: "Nonanticipativity Constraints (NAC): meaning and modeling"
date: "2026-02-12"
week: 7
lesson: 3
slug: "nonanticipativity-constraints-nac-meaning-and-modeling"
---

# Topic: Nonanticipativity Constraints (NAC): meaning and modeling

## 1) Formal definition (what is it, and how can we use it?)

Nonanticipativity constraints (NACs) are restrictions placed on decision variables in stochastic programming models to ensure that decisions made at a particular stage *cannot* depend on information that will only be revealed in the future.  In other words, decisions made *before* a certain realization of uncertainty must be identical across all scenarios that share the same history up to that point.

**Meaning:**

The core idea of NACs is to enforce realistic decision-making under uncertainty. Imagine you're managing an inventory and must place an order *before* knowing the exact demand for next month. The order quantity you choose today *cannot* be a function of information that is only available next month.  You must base your decision solely on available information (e.g., historical demand, forecasts, current inventory levels). NACs mathematically capture this restriction.

**How we can use it:**

*   **Multi-stage decision making:** NACs are crucial in multi-stage stochastic programming problems where decisions are made sequentially over time as uncertainty unfolds.
*   **Ensuring realistic solutions:** By enforcing NACs, the optimization model produces solutions that are implementable in the real world, respecting the temporal flow of information.  Without NACs, the model could find solutions that are optimal in hindsight but impossible to execute.
*   **Decomposition algorithms:** NACs are often the constraints that link different scenarios together, allowing for decomposition algorithms (like Benders decomposition or Lagrangian relaxation) to be applied. By dualizing or relaxing NACs, the problem can be broken down into smaller, more manageable subproblems, one for each scenario. The solutions of these subproblems are then coordinated to satisfy the NACs.

**Mathematical Representation:**

Let:

*   `x_t(ω)` be the decision variable at stage `t` under scenario `ω`.
*   `Ω_t` be the set of possible scenarios up to stage `t`.
*   `ω' ∼ ω` denote that scenarios `ω'` and `ω` share the same history up to stage `t`.

Then, the NAC can be formally expressed as:

`x_t(ω) = x_t(ω')  for all ω, ω' ∈ Ω_t  such that ω' ∼ ω`

This means the decision `x_t` must be the same for all scenarios that have the same history up to time `t`.
## 2) Application scenario

**Scenario:**  A power company needs to plan its electricity generation mix for the next year, considering uncertain electricity demand and uncertain renewable energy generation (e.g., solar and wind).

**Details:**

*   **Stage 1 (Planning):** The power company must decide on the capacity of different power plants (coal, gas, solar, wind) *before* knowing the actual demand or renewable generation. This is a "here-and-now" decision.
*   **Stage 2 (Operation):** After the capacity decisions are made, the actual demand and renewable generation are revealed. The power company then needs to decide how much electricity to generate from each plant to meet the demand. This is a "wait-and-see" decision.

**Nonanticipativity in this context:**

The capacity decisions made in Stage 1 *must* be the same across all scenarios.  The company cannot build different amounts of solar capacity depending on what a crystal ball tells them about future wind conditions. These decisions *must* be made based on the information available *at the time the decision is made*. The operational decisions in Stage 2, however, can vary across scenarios because they are made *after* the uncertainty is revealed.

**Without NACs:** The model might "cheat" by building very little capacity and then relying on extremely favorable renewable energy conditions in some scenarios to cover demand, which is unrealistic.

## 3) Python method (if possible)

Here's how you can model nonanticipativity constraints using the Pyomo optimization library in Python.  This is a simplified example, but demonstrates the key concept.  Assume a two-stage problem.

```python
from pyomo.environ import *

# Number of scenarios
num_scenarios = 3

# Stages
num_stages = 2

# Create a concrete model
model = ConcreteModel()

# Set of scenarios
model.Scenarios = RangeSet(1, num_scenarios)

# Set of stages
model.Stages = RangeSet(1, num_stages)

# Example decision variable (Stage 1, here-and-now)
model.x = Var(within=NonNegativeReals) # Single decision, no scenario index in stage 1

# Example decision variable (Stage 2, wait-and-see)
model.y = Var(model.Scenarios, within=NonNegativeReals) # Stage 2 decision is scenario dependent

# Objective Function (example)
model.objective = Objective(expr= model.x + sum(model.y[s] for s in model.Scenarios), sense=minimize)

# Example constraints (some generic constraints):
model.constraint1 = Constraint(expr = model.x + sum(model.y[s] for s in model.Scenarios) >= 10)

# Nonanticipativity constraints:
# Since x is a single variable defined *outside* of the scenarios, it automatically
# satisfies nonanticipativity for stage 1. If we had multiple stage 1 variables, we would define
# x as Var(model.Scenarios, within=NonNegativeReals)  and then add NACs.

# If we had multiple stage 1 variables indexed by scenarios (which isn't necessary since they're
# all the same), the NACs would look like this (for variables u indexed by scenarios) :

# model.u = Var(model.Scenarios, within=NonNegativeReals)
# def nonanticipativity_rule(model, s):
#     if s > 1:
#         return model.u[s] == model.u[1]  # Force all scenario u values to equal the first scenario u
#     else:
#         return Constraint.Skip #Avoid building redundant constraints
# model.nonanticipativity = Constraint(model.Scenarios, rule=nonanticipativity_rule)

# Solve the model (example)
solver = SolverFactory('glpk') # Or other solver like 'ipopt'
results = solver.solve(model)

# Print results (example)
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found!")
    print(f"x = {model.x.value}")
    for s in model.Scenarios:
        print(f"y[{s}] = {model.y[s].value}")
else:
    print("Solver did not find an optimal solution.")
```

**Explanation:**

*   The code defines decision variables `x` (first-stage, here-and-now) and `y` (second-stage, wait-and-see).
*   Crucially, `x` is *not* indexed by scenario in stage 1.  This implicitly enforces nonanticipativity.  If the first-stage decision *did* depend on the scenario, it would *not* be implementable.  If you *did* have multiple first-stage variables indexed by scenario, you would need explicit NACs as shown in the commented section.
*   The commented code shows the *explicit* form of the NACs, which enforces that a first-stage variable `u` is the same across all scenarios.  This is typically needed when modeling more complex situations where decisions *are* initially indexed by scenario for modeling convenience but must be forced to be identical across scenarios.

## 4) Follow-up question

How do nonanticipativity constraints interact with risk measures in stochastic programming?  Specifically, how does adding risk aversion (e.g., using Conditional Value-at-Risk (CVaR)) to the objective function affect the formulation or implementation of nonanticipativity constraints?
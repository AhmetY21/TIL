---
title: "Nonanticipativity Constraints (NAC): meaning and modeling"
date: "2026-02-12"
week: 7
lesson: 4
slug: "nonanticipativity-constraints-nac-meaning-and-modeling"
---

# Topic: Nonanticipativity Constraints (NAC): meaning and modeling

## 1) Formal definition (what is it, and how can we use it?)

Nonanticipativity constraints (NACs) are a crucial element in stochastic programming, specifically in multi-stage stochastic programming problems. They ensure that decisions made *before* the realization of a future random event (uncertainty) cannot depend on information that is not yet available. In other words, decisions in a particular stage of the model must be the same for all scenarios that have identical histories up to that stage. This reflects the realistic constraint that we can't use future information to decide on actions in the present.

Formally, consider a multi-stage stochastic program with stages *t* = 0, 1, ..., *T*.  Let *x<sub>t</sub>(ω)* represent the decision made at stage *t* under scenario *ω*. The scenario *ω* represents a particular realization of the uncertain parameters over time. The history up to stage *t* for scenario *ω* is the path of the uncertain parameters observed up to stage *t*.

Nonanticipativity constraints enforce that:

*x<sub>t</sub>(ω) = x<sub>t</sub>(ω')*  for all *t*, and for all scenario pairs *(ω, ω')* that have identical histories up to stage *t*.

In simpler terms: If two scenarios have experienced the same events up to stage *t*, then the decision made at stage *t* must be the same in both scenarios.  We "tie" decisions across scenarios that share a common past.

How do we use them?

*   **Ensuring Realism:** They model the fact that decisions cannot be based on future information.
*   **Decomposition:** They enable decomposition algorithms like Benders Decomposition (L-Shaped Method) and Progressive Hedging by providing a way to separate the overall stochastic problem into smaller, more manageable subproblems for each scenario, linked by the nonanticipativity constraints.
*   **Scenario Reduction/Aggregation:** NACs are used to guide scenario reduction techniques, ensuring that the reduced scenario set preserves the essential nonanticipativity properties of the original problem.

## 2) Application scenario

Imagine a power generation company that needs to decide how much electricity to generate from different sources (e.g., coal, natural gas, renewables) over the next few days.  The demand for electricity is uncertain and depends on factors like weather and economic activity.

*   **Stage 0 (Today):** The company decides on the initial generation levels for each power source for the next day.
*   **Stage 1 (Tomorrow):** The actual demand for electricity becomes known, and the company can adjust the generation levels of some sources (e.g., natural gas, which is more flexible) to meet the demand.

Different weather scenarios (e.g., hot weather leading to high demand, mild weather leading to low demand) can unfold. The company needs to make decisions *today* about the initial generation levels, *before* knowing what the actual weather will be *tomorrow*.

Nonanticipativity constraints would be applied at Stage 0. The generation levels decided *today* for coal, gas, and renewables *must be the same* across all weather scenarios because the company doesn't know which scenario will actually occur.  Once the weather is known (Stage 1), the company *can* adjust generation levels based on the specific weather scenario, but *only* for flexible sources like natural gas (if the model allows such adjustments).

Without nonanticipativity constraints, the optimization model could theoretically "peek into the future" and set different initial generation levels for each weather scenario, which is unrealistic.

## 3) Python method (if possible)

Here's how you might model nonanticipativity constraints using the `Pyomo` optimization library in Python. We will use a simplified version of the power generation example.

```python
import pyomo.environ as pyo

# Define the model
model = pyo.ConcreteModel()

# Sets
model.Scenarios = pyo.Set(initialize=['Scenario1', 'Scenario2'])
model.Sources = pyo.Set(initialize=['Coal', 'Gas', 'Renewables'])

# Parameters (simplified, representing initial capacity decisions)
model.capacity_cost = pyo.Param(model.Sources, initialize={'Coal': 10, 'Gas': 5, 'Renewables': 15})

# Variables: capacity of each source in each scenario
model.Capacity = pyo.Var(model.Sources, model.Scenarios, within=pyo.NonNegativeReals)

# Objective (Minimize total cost across scenarios)
def obj_rule(model):
    return sum(model.capacity_cost[s] * model.Capacity[s, scenario] for s in model.Sources for scenario in model.Scenarios)
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


# Nonanticipativity Constraints (NAC) - Enforce same capacity decisions at the beginning
def nac_rule(model, source):
    return model.Capacity[source, 'Scenario1'] == model.Capacity[source, 'Scenario2']
model.nac = pyo.Constraint(model.Sources, rule=nac_rule) #This is the NAC constraint

#Example Constraints
def cap_bounds_rule(model,source,scenario):
  return (0,model.Capacity[source,scenario], 100) # Max capacity of 100 for all sources across all scenarios
model.cap_bounds = pyo.Constraint(model.Sources, model.Scenarios, rule=cap_bounds_rule)

# Solve the model
solver = pyo.SolverFactory('glpk') # You can use other solvers like 'cbc', 'gurobi', 'cplex'
results = solver.solve(model)

# Print the results
if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Solution is optimal")
    for s in model.Sources:
        print(f"Capacity for {s}: Scenario1 = {model.Capacity[s, 'Scenario1'].value}, Scenario2 = {model.Capacity[s, 'Scenario2'].value}")
else:
    print("Solver failed to find an optimal solution.")
    print(results.solver)
```

**Explanation:**

1.  **Model Setup:** We define a simple `Pyomo` model with sets for scenarios (`Scenarios`) and power sources (`Sources`).  We have a variable called `Capacity` that dictates how much capacity of each source to implement for each scenario.
2.  **Nonanticipativity Constraint:** The `nac_rule` function enforces that the capacity decision for each power source must be the same in `Scenario1` and `Scenario2`. This ensures the decision is made without knowing which scenario will occur.  We apply this constraint to all sources using `model.nac = pyo.Constraint(model.Sources, rule=nac_rule)`.
3.  **Objective Function:** The objective is to minimize the total cost of capacity.
4.  **Solver:** We use `glpk` to solve the optimization problem.  You'll likely want to use a more powerful solver like Gurobi or CPLEX for larger, more complex problems.
5.  **Output:** The code prints the optimal capacity levels for each power source and scenario.  You should see that the capacity levels are the same for both scenarios due to the nonanticipativity constraints.

This example demonstrates how to create a nonanticipativity constraint in `Pyomo` by equating decision variables across different scenarios for the initial stage. The core idea is to ensure that decisions are aligned until new information becomes available.

## 4) Follow-up question

How do nonanticipativity constraints affect the complexity of solving a stochastic programming problem, and what techniques are used to mitigate this increased complexity?
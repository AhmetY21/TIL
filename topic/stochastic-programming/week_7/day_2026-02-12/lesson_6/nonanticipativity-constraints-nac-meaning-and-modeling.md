---
title: "Nonanticipativity Constraints (NAC): meaning and modeling"
date: "2026-02-12"
week: 7
lesson: 6
slug: "nonanticipativity-constraints-nac-meaning-and-modeling"
---

# Topic: Nonanticipativity Constraints (NAC): meaning and modeling

## 1) Formal definition (what is it, and how can we use it?)

Nonanticipativity constraints (NACs) are a fundamental concept in multistage stochastic programming.  They enforce the principle that decisions made *before* the realization of some uncertain information (i.e., at a specific stage) cannot *depend* on that future information.  In other words, our first-stage decisions must be the same across all possible scenarios until the uncertainty is revealed. Think of it as a rule that says you can't see the future when making current decisions.

Formally, NACs ensure that decision variables at a given stage are equal across all scenarios that share the same history (information) up to that stage. Let's break that down:

*   **Stages:** Stochastic programs typically involve making decisions over multiple time periods (stages).  Earlier stages occur before later stages.
*   **Scenarios:**  A scenario represents a possible realization of the uncertain parameters (e.g., demand, prices, weather) throughout all stages.
*   **History:** The history of a scenario up to a given stage is the realization of the uncertain parameters up to that stage. Scenarios with the same history up to stage *t* are indistinguishable from the decision-maker's perspective at stage *t*.

Therefore, for any two scenarios *s* and *s'* that have the same history up to stage *t*, the decisions made at stage *t* must be the same:

```
x_t(s) = x_t(s')  for all scenarios s, s' such that history_t(s) = history_t(s')
```

Where:

*   `x_t(s)` is the decision variable at stage *t* in scenario *s*.
*   `history_t(s)` represents the history (information) of scenario *s* up to stage *t*.

**How can we use it?**

NACs are crucial for building realistic and implementable stochastic programming models.  Without them, the model might prescribe decisions that are optimal *in hindsight* (knowing the future), but impossible to execute in reality.  By enforcing nonanticipativity, we ensure that the solution is feasible and adaptable to different scenario realizations as they unfold over time. NACs transform a theoretical optimal solution into a practically applicable policy. They are especially relevant when modeling sequential decision making under uncertainty such as inventory management, power systems operation, and financial portfolio optimization.

## 2) Application scenario

Consider a supply chain management problem where a company needs to decide how much inventory to order each month to meet uncertain demand. The company operates for three months (stages). Demand is uncertain and can follow two scenarios: high demand or low demand. In the first month, the company doesn't know what scenario will occur. In the second month, the demand scenario is revealed.

*   **Stage 1 (Month 1):** The company must decide how much to order *before* knowing the actual demand scenario (high or low).
*   **Stage 2 (Month 2):** The demand scenario is revealed. The company can then adjust its ordering decision based on the realized demand (either high or low).
*   **Stage 3 (Month 3):** Final demand realization.

Let `x_t(s)` be the order quantity in month *t* under scenario *s*.

Without NACs, the model *could* potentially choose different order quantities in month 1 for the "high demand" and "low demand" scenarios, even though the company doesn't know which scenario will occur in month 1.  This is unrealistic.

To enforce nonanticipativity, we would impose the following constraint:

```
x_1(high_demand_scenario) = x_1(low_demand_scenario)
```

This constraint ensures that the order quantity in month 1 is the same, regardless of whether the scenario eventually turns out to be high demand or low demand. In month 2, however, the order quantity *can* be different depending on the revealed demand scenario, because at that point the company knows the actual demand.

In this scenario, NACs are essential to ensure that the ordering decisions are implementable in practice.  The company can't order different quantities based on future knowledge it doesn't have at the time of ordering.

## 3) Python method (if possible)

Here's an example using Pyomo to model the above supply chain problem and include NACs:

```python
from pyomo.environ import *

# Define the model
model = ConcreteModel()

# Set of scenarios
model.Scenarios = Set(initialize=['high_demand', 'low_demand'])

# Set of stages
model.Stages = Set(initialize=[1, 2, 3])

# Parameters
demand = {
    ('high_demand', 1): 50,
    ('high_demand', 2): 60,
    ('high_demand', 3): 70,
    ('low_demand', 1): 30,
    ('low_demand', 2): 40,
    ('low_demand', 3): 50
}

model.Demand = Param(model.Scenarios, model.Stages, initialize=demand)

# Decision Variables
model.OrderQuantity = Var(model.Scenarios, model.Stages, domain=NonNegativeReals)

# Objective function
def objective_rule(model):
    cost = 0
    for s in model.Scenarios:
        for t in model.Stages:
            cost += model.OrderQuantity[s, t]  # Placeholder cost - modify as needed
    return cost

model.Objective = Objective(rule=objective_rule, sense=minimize)


# Nonanticipativity Constraint
def nac_rule(model):
    return model.OrderQuantity['high_demand', 1] == model.OrderQuantity['low_demand', 1]

model.NAC = Constraint(rule=nac_rule)


# Inventory Balance Constraints (Example - needs to be adapted to full problem)
def inventory_balance_rule(model, s, t):
    if t == 1:
        inventory = model.OrderQuantity[s, t] - model.Demand[s, t]
    else:
        inventory = model.OrderQuantity[s, t] + model.OrderQuantity[s, t-1] - model.Demand[s, t]
    return inventory >= 0
model.InventoryBalance = Constraint(model.Scenarios, model.Stages, rule=inventory_balance_rule)

# Solve the model
solver = SolverFactory('glpk')  # Or another solver like 'ipopt'
results = solver.solve(model)

# Print results
if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("Solution is optimal")
    for s in model.Scenarios:
        for t in model.Stages:
            print(f"Order Quantity in scenario {s}, stage {t}: {model.OrderQuantity[s, t].value}")
else:
    print("Solver failed")

```

**Explanation:**

1.  **Import Pyomo:** Imports the necessary library.
2.  **Define Model:**  Creates a `ConcreteModel`.
3.  **Sets:** Defines the set of scenarios (`Scenarios`) and stages (`Stages`).
4.  **Parameters:** Defines the `Demand` parameter for each scenario and stage.
5.  **Decision Variable:**  `OrderQuantity` is a variable that determines the amount ordered.
6.  **Objective Function:** Defines the objective function (currently a placeholder -- you'll need to adapt it to your specific cost structure).  This minimizes some function of order quantity, inventory, or whatever is relevant in the problem.
7.  **Nonanticipativity Constraint (NAC):** The `nac_rule` function enforces that the order quantity in stage 1 is the same across all scenarios. This is the key part demonstrating NAC implementation.  `model.NAC = Constraint(rule=nac_rule)` adds this as a constraint to the Pyomo model.  The specific NAC added is just for month 1, but other NACs can be added at later stages, depending on what information is known when.
8.  **Inventory Balance Constraints:** `inventory_balance_rule` defines simple inventory balance constraint, where inventory must be positive. This ensures feasibility.
9.  **Solve and Print:** Solves the model using GLPK solver (you might need to install it) and prints the optimal order quantities.

This example shows a basic implementation of NACs. You'll likely need to adapt it to fit the specific details of your stochastic programming problem, including defining appropriate objective functions, and adding more constraints and variables.

## 4) Follow-up question

How do Nonanticipativity Constraints change when dealing with a scenario tree with more complex branching structures (e.g., three possible outcomes at stage 1 instead of just two)? How does this affect the computational complexity of solving the stochastic program?
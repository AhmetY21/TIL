---
title: "Deterministic vs Stochastic Models: recourse, adaptivity, and information"
date: "2026-02-11"
week: 7
lesson: 3
slug: "deterministic-vs-stochastic-models-recourse-adaptivity-and-information"
---

# Topic: Deterministic vs Stochastic Models: recourse, adaptivity, and information

## 1) Formal definition (what is it, and how can we use it?)

This topic concerns the fundamental distinction between deterministic and stochastic optimization models, focusing on how they handle uncertainty, and specifically addressing the concepts of recourse, adaptivity, and the role of information.

**Deterministic Models:** These models assume all parameters are known and fixed at the time of optimization. The solution is a single, pre-determined plan executed regardless of what actually happens. They ignore uncertainty. The optimization problem is typically of the form:

`minimize f(x)`
`subject to g(x) <= 0`
`x ∈ X`

where `x` is the decision variable, `f` is the objective function, `g` represents constraints, and `X` is the feasible set.  All coefficients in `f` and `g` are known constants.

**Stochastic Models:** These models explicitly account for uncertainty in some parameters, which are represented as random variables. The goal is to find a solution that is robust (i.e., performs well) across a range of possible outcomes. Stochastic Programming (SP) offers a framework to model and solve these problems.

**Recourse:**  This refers to the ability to take corrective actions after the uncertainty is revealed. In a *two-stage* stochastic program, for instance, we make an initial (here-and-now) decision `x` before observing the realization of the random variable `ξ`.  After observing `ξ`, we can take a second-stage decision (recourse action) `y(x, ξ)` to mitigate the effects of the uncertainty. The recourse decision `y` depends on both the initial decision `x` and the realized value of the random variable `ξ`.

**Adaptivity:** Adaptivity is directly related to recourse. It is the capability of the model to adjust its decisions based on the observed values of the uncertain parameters. The recourse decisions `y(x, ξ)` represent the adaptive component of the solution.  Without recourse, there is no adaptivity; the initial decision `x` has to be good for all realizations of `ξ`.  Multi-stage stochastic programs extend this concept to multiple periods, allowing decisions to adapt sequentially as information unfolds.

**Information:** The information structure dictates how and when we learn about the uncertain parameters.  In two-stage problems, all uncertainty is revealed between the first and second stages.  In multi-stage problems, information is revealed gradually over time. The "non-anticipativity" constraints in multi-stage stochastic programming are crucial; they ensure that decisions made at a given stage can only depend on the information available up to that stage, not on future information. The quality of the solution crucially depends on the accurate representation of the probability distribution that drives the realization of uncertain parameters.

**How can we use this?** Understanding these concepts allows us to choose the appropriate modeling approach. If the impact of uncertainty is negligible or if quick solutions are paramount, a deterministic model might suffice. If uncertainty is significant and corrective actions are possible, a stochastic model with recourse is more appropriate. The ability to model adaptivity based on revealed information allows for creating more robust and realistic plans. Different information structures can be modeled in multistage problems allowing for modelling complex scenarios.

## 2) Application scenario

**Scenario: Supply Chain Planning under Demand Uncertainty**

Consider a company that needs to decide how much inventory to order (`x`, first-stage decision) before knowing the actual demand for its product (`ξ`, a random variable). If the company orders too little, it will lose sales due to stockouts. If it orders too much, it will incur holding costs.  After observing the actual demand, the company can take recourse actions (`y(x, ξ)`):

*   **If demand is higher than expected:**  The company might expedite production, ship from another warehouse, or backorder.
*   **If demand is lower than expected:** The company might offer discounts, store excess inventory, or return goods to suppliers.

A deterministic model would simply use an average demand value, leading to potentially poor decisions if the actual demand deviates significantly from the average. A stochastic model with recourse allows the company to optimize its inventory levels while considering the different scenarios of demand and the corresponding recourse actions to minimize the overall cost (ordering cost + holding cost + stockout cost + recourse cost).

In this scenario, *recourse* is the actions the company takes after demand is known. *Adaptivity* is the company's ability to change its response based on actual demand. *Information* is the realization of the demand, which is revealed after the initial inventory decision is made.

## 3) Python method (if possible)

While specific libraries for large-scale stochastic programming vary, we can illustrate the concept with a simplified example using `pyomo` and a simple scenario approach for representing the uncertainty.

```python
from pyomo.environ import *
import numpy as np

# Define the model
model = ConcreteModel()

# Parameters
holding_cost = 1
stockout_cost = 5

# Scenarios for demand
scenarios = [50, 80, 120]
probabilities = [0.3, 0.4, 0.3]

# First-stage decision variable (order quantity)
model.order_quantity = Var(domain=NonNegativeReals)

# Second-stage decision variables (recourse actions)
model.inventory = Var(scenarios, domain=NonNegativeReals)  # Inventory left
model.shortage = Var(scenarios, domain=NonNegativeReals) # Shortage

# Objective function (expected cost)
def objective_rule(model):
    return (model.order_quantity * 2 + # ordering cost
            sum(probabilities[i] * (holding_cost * model.inventory[scenarios[i]] +
                                  stockout_cost * model.shortage[scenarios[i]])
                for i in range(len(scenarios))))
model.objective = Objective(rule=objective_rule, sense=minimize)


# Constraints (balance constraints for each scenario)
def demand_balance_rule(model, scenario):
    return model.order_quantity == model.inventory[scenario] + model.shortage[scenario] + scenario
model.demand_balance = Constraint(scenarios, rule=demand_balance_rule)


# Solve the model
solver = SolverFactory('glpk') # Or any other suitable solver
solver.solve(model)

# Print the results
print(f"Optimal order quantity: {model.order_quantity.value}")
for scenario in scenarios:
    print(f"Scenario {scenario}: Inventory={model.inventory[scenario].value}, Shortage={model.shortage[scenario].value}")
```

This example defines a two-stage stochastic program.  The `model.order_quantity` is the first-stage decision.  `model.inventory` and `model.shortage` are the recourse variables, which are scenario-dependent. The `demand_balance_rule` demonstrates adaptivity: the amount of inventory and shortage adjusts depending on the realized demand in each scenario.

**Important notes:**

*   This is a simplified scenario-based approach.  For continuous distributions, sampling techniques (like Monte Carlo) are used.
*   More advanced libraries like `pysp` (part of Pyomo) and dedicated stochastic programming solvers (e.g., Gurobi with its scenario features) are needed for larger and more complex problems.
*   The choice of solver is crucial; `glpk` is used here for simplicity, but commercial solvers like Gurobi or CPLEX are often necessary for real-world applications.

## 4) Follow-up question

How does the choice of the probability distribution used to model the uncertainty influence the solution of a stochastic program, and what are the potential pitfalls of using an inaccurate or poorly estimated distribution?
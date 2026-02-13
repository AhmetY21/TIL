---
title: "Nonanticipativity Constraints (NAC): meaning and modeling"
date: "2026-02-13"
week: 7
lesson: 1
slug: "nonanticipativity-constraints-nac-meaning-and-modeling"
---

# Topic: Nonanticipativity Constraints (NAC): meaning and modeling

## 1) Formal definition (what is it, and how can we use it?)

**What is it?**

Nonanticipativity constraints (NACs) are a fundamental concept in multi-stage stochastic programming. They enforce the requirement that decisions made *before* the realization of future uncertainty cannot depend on the *specific* realization of that uncertainty.  In simpler terms, your first-stage decision cannot *anticipate* what will happen later.  Decisions must be based only on the information available at the time they are made.

**How can we use it?**

NACs are crucial for formulating realistic stochastic optimization problems because they prevent unrealistic or impossible solutions. Imagine making an investment decision *today* based on knowing the *exact* weather conditions three months from now. That's clearly impossible. NACs ensure that your model only considers information actually available at each stage. They are essential to correctly model decisions that must be made before all the uncertainty is revealed.

Mathematically, NACs often take the form of equality constraints that ensure the first-stage (or generally, *t*-th stage) decision variables are equal across different scenarios (or realizations) of the uncertainty *up to time t*.  Let's say `x_t` represents the decision at time `t`, and `ω` represents a scenario.  Then, for two scenarios `ω_1` and `ω_2` that are indistinguishable up to time `t`, the NAC would be:

`x_t(ω_1) = x_t(ω_2)` for all scenarios `ω_1`, `ω_2` where the information available at time `t` is the same.

This constraint ensures that the decision `x_t` is the same regardless of which of those indistinguishable scenarios actually occurs.  Essentially, the first-stage decision is only one decision that applies to all scenarios.

## 2) Application scenario

Consider an electric utility company planning its electricity generation capacity. They need to decide how much to invest in different types of power plants (solar, wind, gas) today (stage 1).  The future electricity demand (stage 2) is uncertain, and depends on factors like economic growth and weather conditions.

*   **Decisions:**
    *   `x`: Capacity of each type of power plant to build (stage 1 - made now).  This is a vector of decisions.
    *   `y(ω)`: Amount of electricity to generate from each plant type under scenario `ω` (stage 2 - made after demand is revealed).  This is a vector of decisions for each scenario.
*   **Uncertainty:**
    *   `ω`: Scenarios representing different possible electricity demand patterns.

Without NACs, the model might allow the company to choose different capacity investments for each scenario, essentially *knowing* in advance which demand scenario will occur. This is not realistic.

The NACs would require:

`x(ω_1) = x(ω_2)`  for all scenarios `ω_1`, `ω_2`.

This ensures that the capacity investment decision `x` is the *same* regardless of the future demand scenario. The company makes one capacity investment decision that must be suitable for *all* plausible demand scenarios. Later, after observing the actual demand scenario, the company can then adjust the generation levels `y(ω)` to meet that demand.

## 3) Python method (if possible)

Here's an example using the Pyomo modeling language to demonstrate how nonanticipativity constraints can be implemented.  This example uses a simple two-stage model.

```python
from pyomo.environ import *

# Define a model
model = ConcreteModel()

# Define scenarios
scenarios = ['low_demand', 'high_demand']
model.Scenarios = Set(initialize=scenarios)

# Define first-stage decision variable (investment capacity - same across scenarios)
model.Investment = Var(domain=NonNegativeReals)

# Define second-stage decision variables (production level - different for each scenario)
model.Production = Var(model.Scenarios, domain=NonNegativeReals)

# Define scenario probabilities (assuming equal probability for simplicity)
scenario_probabilities = {'low_demand': 0.5, 'high_demand': 0.5}
model.ScenarioProb = Param(model.Scenarios, initialize=scenario_probabilities)


# Define objective function (expected cost, simplified)
def objective_rule(model):
    return sum(model.ScenarioProb[s] * (model.Production[s] - 0.1*model.Investment)**2 for s in model.Scenarios) #simplified cost
model.Objective = Objective(rule=objective_rule, sense=minimize)


# Define a simplified production constraint (cannot exceed installed capacity, plus some demand factor)
def production_constraint_rule(model, s):
    demand_factor = 0.5 if s == 'low_demand' else 1.0 # Example demand factors
    return model.Production[s] <= model.Investment + demand_factor
model.ProductionConstraint = Constraint(model.Scenarios, rule=production_constraint_rule)


# **Define Nonanticipativity Constraints**  (here, trivial since only one first-stage var)
# In this particular example, because 'Investment' is a single variable and designed to be identical across all scenarios *by construction*, the constraint isn't strictly *necessary*.
#However, if 'Investment' *was* indexed by the scenarios, then we would *require* this constraint:

def nonanticipativity_rule(model, s):
    return model.Investment == model.Investment
    #More generally (if 'Investment' were indexed) something like:
    #return model.Investment[s] == model.Investment[scenarios[0]] #forcing all investments equal to the first scenario's value

model.Nonanticipativity = Constraint(model.Scenarios, rule=nonanticipativity_rule) #Add the constraint to the model


# Solve the model
solver = SolverFactory('ipopt') # You might need to install an appropriate solver
results = solver.solve(model)

# Print the results
print("Investment:", model.Investment.value)
for s in scenarios:
    print(f"Production in {s}:", model.Production[s].value)
```

**Explanation:**

*   The key part is the `nonanticipativity_rule`.  In this simple example, since there is only one first-stage variable ('Investment') and it is NOT indexed by scenario, the NAC effectively does nothing. But I've included it to demonstrate the *structure* of how you *would* define a NAC. If your first-stage variables *were* indexed by scenario (which would technically allow them to vary across scenarios without a NAC), then the NAC constraint would be crucial to enforce equality between these variables across all scenarios. The more general form, commented out, shows how one could anchor to one of the scenarios for comparison.
*   The production constraint demonstrates how second-stage decisions can vary based on the scenario.
*  The `ScenarioProb` parameter represents the probability of each scenario occurring.
*   The objective function represents the expected cost across all scenarios.

This Pyomo example demonstrates the basic structure.  In more complex models, you would adapt the `nonanticipativity_rule` to enforce equality of the *relevant* first-stage (or *t*-th stage) decisions across scenarios that are indistinguishable up to that stage.

## 4) Follow-up question

How do you handle a very large number of scenarios in stochastic programming? Specifically, what are some common techniques to reduce the computational burden of models with numerous scenarios, while still accurately representing the uncertainty? Consider scenario reduction techniques, sampling methods, and approximation strategies.
---
title: "Modeling Stochastic Programs in Pyomo/PySP: scenarios + nonanticipativity"
date: "2026-02-18"
week: 8
lesson: 6
slug: "modeling-stochastic-programs-in-pyomo-pysp-scenarios-nonanticipativity"
---

# Topic: Modeling Stochastic Programs in Pyomo/PySP: scenarios + nonanticipativity

## 1) Formal definition (what is it, and how can we use it?)

In stochastic programming, we deal with optimization problems where some of the parameters are uncertain and represented by random variables.  "Modeling Stochastic Programs in Pyomo/PySP: scenarios + nonanticipativity" refers to a specific approach for representing and solving these problems using Pyomo (a Python optimization modeling language) and PySP (Pyomo's stochastic programming extension).

**Scenarios:** The fundamental idea is to discretize the uncertainty by representing the random variables with a finite set of possible realizations, called scenarios.  Each scenario represents a possible "future" or "world" that might unfold.  Each scenario is associated with a probability of occurrence.  We then optimize over *all* scenarios, taking their probabilities into account.

**Nonanticipativity:** This is a crucial concept. It means that decisions made *before* the uncertainty is revealed (first-stage decisions) must be the same across all scenarios.  Think of it as: you can't base your upfront decisions on knowing the future, because you don't know which scenario will actually occur. Later-stage decisions (made after some uncertainty is resolved) *can* depend on the scenario.  The nonanticipativity constraints enforce this.

**How can we use it?**  This approach allows us to:

*   Handle optimization problems with uncertainty.
*   Make robust decisions that perform well across a range of possible future outcomes.
*   Model two-stage (or multi-stage) decision-making processes where decisions are made sequentially as information is revealed.
*   Solve stochastic problems by converting them into deterministic equivalent problems that can be handled by standard optimization solvers.
*   Use decomposition techniques (like Progressive Hedging in PySP) to solve large-scale stochastic programs.

In summary, scenarios provide a discrete representation of uncertainty, and nonanticipativity ensures that decisions made before observing the outcome of the uncertainty are consistent across all potential future realities.

## 2) Application scenario

Consider a supply chain management problem. A company needs to decide how much inventory to order *now* (first-stage decision) to meet future demand.  The future demand is uncertain and can be represented by different scenarios (e.g., "high demand," "medium demand," "low demand").

*   **First-stage decision (x):**  The quantity of inventory to order. This *must* be the same across all scenarios due to nonanticipativity.  You can't order a different amount of inventory based on what the future demand *might* be.

*   **Uncertain parameter (d_s):** Demand in scenario 's'.

*   **Second-stage decision (y_s):** The amount of product to sell in scenario 's'. This can vary between scenarios. You can sell more when demand is high and less when demand is low.

*   **Objective:** Minimize the total cost, including inventory holding costs, order costs, and potential lost sales costs.

The nonanticipativity constraint forces the inventory order quantity (x) to be the same regardless of which demand scenario ultimately occurs. This is a realistic and essential constraint in many real-world supply chain problems.

## 3) Python method (if possible)

```python
from pyomo.environ import *
from pyomo.core import *
from pyomo.pysp.scenariotree.tree_model import CreateAbstractScenarioTreeModel

# Define the abstract model
model = AbstractModel()

# Scenario set
model.Scenarios = Set()

# Parameters (indexed by scenario)
model.Demand = Param(model.Scenarios)

# First-stage variable (same for all scenarios)
model.OrderQuantity = Var(domain=NonNegativeReals)

# Second-stage variable (scenario-dependent)
model.Sales = Var(model.Scenarios, domain=NonNegativeReals)

# Objective function
def obj_rule(model):
    return model.OrderQuantity + sum(model.Sales[s] for s in model.Scenarios)
model.obj = Objective(rule=obj_rule, sense=minimize)

# Constraint: Sales cannot exceed Demand
def sales_limit_rule(model, s):
    return model.Sales[s] <= model.Demand[s]
model.SalesLimit = Constraint(model.Scenarios, rule=sales_limit_rule)

# Nonanticipativity constraint (implicit in this case, as OrderQuantity is defined once)
# If OrderQuantity was defined as indexed by scenario initially, you would need to add:
# def nonant_rule(model, s1, s2):
#     return model.OrderQuantity[s1] == model.OrderQuantity[s2]
# model.NonAnt = Constraint(model.Scenarios, model.Scenarios, rule = nonant_rule)

# Create scenario data
data = {None: {
    'Scenarios': {'set': ['Scenario1', 'Scenario2']},
    'Demand': {'Scenario1': 10, 'Scenario2': 20}
}}

# Create an instance of the model
instance = model.create_instance(data=data)

# Solve the model
opt = SolverFactory('glpk')  # or any other solver
results = opt.solve(instance)

# Print the results
print("Order Quantity:", value(instance.OrderQuantity))
print("Sales in Scenario 1:", value(instance.Sales['Scenario1']))
print("Sales in Scenario 2:", value(instance.Sales['Scenario2']))

# Scenario tree (for PySP) - Required for more advanced features
scenario_tree = CreateAbstractScenarioTreeModel()

# Define nodes
scenario_tree.Stages.add(1)
scenario_tree.Stages.add(2)
scenario_tree.Nodes.add('Root')
scenario_tree.Nodes.add('Scenario1')
scenario_tree.Nodes.add('Scenario2')

# Define stage of each node
scenario_tree.Stage.add_attribute('Root', 1)
scenario_tree.Stage.add_attribute('Scenario1', 2)
scenario_tree.Stage.add_attribute('Scenario2', 2)

# Define children of each node
scenario_tree.Children.add_attribute('Root', ['Scenario1','Scenario2'])

# Define probability of each scenario
scenario_tree.ConditionalProbability.add_attribute('Scenario1', 0.5)
scenario_tree.ConditionalProbability.add_attribute('Scenario2', 0.5)

# Link scenario data to scenarios
scenario_tree.Scenarios.add('Scenario1')
scenario_tree.Scenarios.add('Scenario2')

scenario_tree.ScenarioInstances.add('Scenario1', 'Scenario1')
scenario_tree.ScenarioInstances.add('Scenario2', 'Scenario2')

# Create instance of the scenario tree
scenario_tree_instance = scenario_tree.create_instance()

```

**Explanation:**

1.  We define an `AbstractModel` in Pyomo.
2.  `model.Scenarios` is a set representing the different scenarios.
3.  `model.Demand` is a parameter representing the demand in each scenario.
4.  `model.OrderQuantity` is the first-stage variable (inventory to order). It is *not* indexed by scenario, thus implicitly enforcing nonanticipativity. If we initially made it indexed by scenario and intended it to be a first-stage variable, we would need to add an explicit nonanticipativity constraint, like the commented-out `model.NonAnt` constraint.
5.  `model.Sales` is the second-stage variable (quantity to sell) and is indexed by scenario, allowing it to vary.
6.  The objective is to minimize the total cost (simplified here).
7.  The `SalesLimit` constraint ensures that sales do not exceed demand.
8.  The scenario data is provided in the `data` dictionary.
9.  An instance of the model is created and solved using `glpk`.
10. The final section creates a scenario tree model for use with PySP. It defines the stages, nodes, children, conditional probabilities, scenario instances, and links the scenario data to the scenarios. This structure is crucial for using PySP's advanced features, such as decomposition algorithms.

## 4) Follow-up question

How would the model and code change if we wanted to model a three-stage stochastic program where, after observing the first-stage demand, we make a decision on whether to increase production capacity, and *then* after a second period of uncertainty we make a final sales decision? How would we incorporate the production capacity decision variable and its associated constraints into the model?
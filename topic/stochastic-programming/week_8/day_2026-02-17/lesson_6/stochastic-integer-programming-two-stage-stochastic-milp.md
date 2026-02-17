---
title: "Stochastic Integer Programming: two-stage stochastic MILP"
date: "2026-02-17"
week: 8
lesson: 6
slug: "stochastic-integer-programming-two-stage-stochastic-milp"
---

# Topic: Stochastic Integer Programming: two-stage stochastic MILP

## 1) Formal definition (what is it, and how can we use it?)

Two-stage stochastic mixed-integer linear programming (two-stage stochastic MILP) is a type of stochastic programming problem that deals with decision-making under uncertainty, where decisions are made in two stages. The key characteristic is that some of the decision variables are integer-valued, making the problem significantly more challenging to solve than its continuous counterpart.

* **What is it?**
    * **First-stage (Here-and-Now) Decisions:** These decisions must be made *before* the uncertainty is revealed. Think of them as strategic, long-term decisions. They are represented by variables, say, `x`. These are generally (but not always) integer or binary variables in a MILP setting.
    * **Second-stage (Wait-and-See) Decisions:** These decisions can be made *after* the uncertainty is revealed. They are corrective or operational decisions, adjusting to the realized scenario. They are represented by variables, say, `y(ω)`. Note that the second-stage decisions `y` are scenario-dependent and thus are functions of the scenario `ω`. These can also be integer or continuous.
    * **Uncertainty:** The uncertainty is modeled as a finite set of scenarios, each with an associated probability of occurrence. A scenario `ω` represents a specific realization of the uncertain parameters (e.g., demand, cost, etc.).
    * **Objective Function:** The objective function aims to minimize the sum of the first-stage cost and the expected value of the second-stage cost across all scenarios.  In general form:

    ```
    min  c'x + E[Q(x, ω)]
    s.t. Ax <= b
         x ∈ X
    ```

    Where:
        * `x` are the first-stage decisions.
        * `c` is the cost vector for the first-stage decisions.
        * `E[Q(x, ω)]` is the expected second-stage cost. `Q(x, ω)` is the optimal value of the second-stage problem given first stage decisions `x` and scenario `ω`.
        * `Ax <= b` are first-stage constraints.
        * `X` is the feasible region for the first-stage decisions (often defined by integrality constraints on `x`).

    The second-stage problem for a given scenario ω is:

    ```
    Q(x, ω) = min q(ω)'y(ω)
            s.t. T(ω)x + W(ω)y(ω) <= h(ω)
                 y(ω) ∈ Y
    ```

    Where:
        * `y(ω)` are the second-stage decisions for scenario `ω`.
        * `q(ω)` is the cost vector for the second-stage decisions under scenario `ω`.
        * `T(ω)x + W(ω)y(ω) <= h(ω)` are the second-stage constraints under scenario `ω` that link the first-stage and second-stage decisions.
        * `Y` is the feasible region for the second-stage decisions (often defined by integrality constraints on `y`).
* **How can we use it?**
    * **Modeling Complex Problems:** It is useful for modeling problems where some decisions must be made before knowing all the information, and then corrective actions can be taken once more information is available.
    * **Risk Management:** By considering multiple scenarios, the model explicitly accounts for uncertainty, which can lead to more robust solutions.
    * **Optimization Under Uncertainty:**  Provides a framework for finding solutions that are optimal on average across a range of possible outcomes.
    * **Capacity Planning:**  Useful for determining the optimal capacity of resources given uncertain future demand.

## 2) Application scenario

**Example: Supply Chain Design with Demand Uncertainty**

A company needs to decide on the number of warehouses to build (first-stage decisions) *before* knowing the actual demand in different regions. After the warehouses are built, the company can decide how much product to ship from each warehouse to each region (second-stage decisions) based on the realized demand.

* **First-stage decisions (x):**
    * Binary variables: `x_i = 1` if warehouse `i` is built, `0` otherwise.
* **Second-stage decisions (y(ω)):**
    * Continuous variables: `y_{ij}(ω)` = amount of product shipped from warehouse `i` to region `j` under scenario `ω`.
* **Uncertainty (ω):**
    * Scenarios represent different possible demand levels in each region. Each scenario has an associated probability.
* **Objective:**
    * Minimize the cost of building warehouses plus the expected cost of shipping products across all demand scenarios.
* **Constraints:**
    * First-stage constraints: Limit on the total number of warehouses that can be built, budget constraints.
    * Second-stage constraints:  Demand must be met in each region for each scenario. Warehouse capacity limits. Flow balance constraints.

In this example, the company makes strategic decisions about warehouse location upfront and then makes operational shipping decisions after the demand uncertainty is resolved.  The stochastic MILP model helps the company find the best warehouse configuration that minimizes costs while ensuring that customer demand can be satisfied under various possible demand scenarios.

## 3) Python method (if possible)

Here's an example using the `pyomo` library. Note: This is a simplified example for illustrative purposes. Building a full-scale model can be more complex.  This example models a simple capacity planning problem, where the first stage is about the amount of a factory to build and the second stage is how much to produce depending on the demand.

```python
from pyomo.environ import *
import numpy as np

# Number of scenarios
num_scenarios = 3

# Scenario probabilities
probabilities = {1: 0.3, 2: 0.4, 3: 0.3}

# Demand scenarios
demand = {1: 10, 2: 15, 3: 20}

# Production cost per unit
production_cost = 2

# Capacity cost per unit of capacity built
capacity_cost = 5

# Create a concrete model
model = ConcreteModel()

# First-stage variable: Capacity to build
model.capacity = Var(domain=NonNegativeReals)  # Continuous variable

# Second-stage variables: Production level for each scenario
model.production = Var(range(1, num_scenarios + 1), domain=NonNegativeReals)

# Objective function
def objective_rule(model):
    return capacity_cost * model.capacity + sum(probabilities[s] * production_cost * model.production[s] for s in range(1, num_scenarios + 1))
model.objective = Objective(rule=objective_rule, sense=minimize)

# First-stage constraint:  Capacity Limit (optional, can be omitted)
# model.capacity_limit = Constraint(expr=model.capacity <= 100)  # Example upper bound

# Second-stage constraints: Production must be less than or equal to capacity AND production must meet demand
def production_capacity_rule(model, s):
    return model.production[s] <= model.capacity
model.production_capacity = Constraint(range(1, num_scenarios + 1), rule=production_capacity_rule)


def production_demand_rule(model, s):
    return model.production[s] >= demand[s]
model.production_demand = Constraint(range(1, num_scenarios + 1), rule=production_demand_rule)


# Solve the model
solver = SolverFactory('glpk') # Or 'gurobi', 'cplex', etc.  glpk is a free solver.
solver.solve(model)

# Print the results
print("Optimal capacity:", model.capacity.value)
for s in range(1, num_scenarios + 1):
    print("Production in scenario", s, ":", model.production[s].value)
print("Objective value:", model.objective.expr()) # model.objective.value also works
```

**Explanation:**

1. **Scenario Definition:** Defines the number of scenarios, their probabilities, and the demand for each scenario.
2. **Model Creation:** Creates a `ConcreteModel` in Pyomo.
3. **Variables:** Defines the first-stage variable `capacity` (continuous) and the second-stage variable `production` for each scenario.
4. **Objective Function:** Minimizes the sum of the capacity cost and the expected production cost.
5. **Constraints:**
   *  `production_capacity`:  Ensures that production in each scenario does not exceed the built capacity.
   * `production_demand`: Ensures demand is met in each scenario.
6. **Solving:**  Uses the `glpk` solver (a free solver) to solve the model. You can replace it with `gurobi` or `cplex` if you have a license for them.
7. **Output:** Prints the optimal capacity and production levels for each scenario.

**Important Considerations:**

* **Integer Variables:** If the first-stage or second-stage variables were integer, you would set `domain=Integers` or `domain=Binary` in the `Var` definition.  The solver choice (`gurobi`, `cplex`, etc.) becomes even more critical when dealing with integer variables.
* **Solver Choice:**  Solvers like Gurobi or CPLEX are generally much more efficient for solving MILPs than open-source solvers like GLPK.
* **Scenario Generation:**  The quality of the stochastic programming solution depends heavily on the quality and representativeness of the scenarios. Scenario generation techniques (e.g., sampling, scenario reduction) are important in practice.

## 4) Follow-up question

How does the complexity of solving a two-stage stochastic MILP change as the number of scenarios increases, and what techniques can be used to mitigate this?
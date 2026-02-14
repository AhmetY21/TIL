---
title: "Computational Challenge: size blow-up and why decomposition is needed"
date: "2026-02-14"
week: 7
lesson: 1
slug: "computational-challenge-size-blow-up-and-why-decomposition-is-needed"
---

# Topic: Computational Challenge: size blow-up and why decomposition is needed

## 1) Formal definition (what is it, and how can we use it?)

In stochastic programming, we deal with optimization problems where some parameters are uncertain and represented by probability distributions.  A common approach is to discretize the uncertainty, creating a *scenario tree* or *scenario set*. Each scenario represents a possible realization of the uncertain parameters.  Solving the problem directly involves creating a deterministic equivalent model that incorporates all possible scenarios.

The "size blow-up" refers to the exponential increase in the size (number of variables and constraints) of the deterministic equivalent model as the number of scenarios and/or the number of stages in the stochastic program grows. Imagine we have a stochastic program with *T* stages (time periods) and *S* scenarios per stage. In a scenario tree, the total number of scenarios at the final stage *T* is *S<sup>T</sup>*. If each scenario requires storing a decision vector of size *n*, the total size of the decision variables can become prohibitively large even for moderate values of *S* and *T*.

Formally, if we have a multi-stage stochastic program with recourse, the deterministic equivalent formulation links decisions across all stages and scenarios. A typical formulation might look something like this conceptually (without explicitly writing out the complex matrices):

Minimize:  Expected cost over all scenarios

Subject to:
*   First-stage decisions
*   Constraints linking first-stage decisions to second-stage decisions under each scenario
*   Constraints linking second-stage decisions to third-stage decisions under each scenario, and so on.

The key here is that the constraints at later stages *replicate* for each scenario. This replication leads to the exponential growth in the size of the model.

**Why is decomposition needed?** Because the size blow-up makes solving the deterministic equivalent model intractable for many realistic problems. Decomposition methods aim to break down the large problem into smaller, more manageable subproblems that can be solved iteratively.  They leverage the structure of the problem, such as the separability of scenarios and the link between stages, to reduce the computational burden.  Common decomposition techniques include:

*   **Benders Decomposition (L-Shaped Method):** Iteratively adds cuts (linear approximations) to the master problem based on information from solving subproblems (typically scenarios or stages).
*   **Stochastic Dual Dynamic Programming (SDDP):** Extends Benders decomposition to multi-stage problems, using backward recursion to approximate the cost-to-go function (future costs) at each stage.
*   **Progressive Hedging (PH):** Encourages scenarios to converge towards a common first-stage decision by introducing penalty terms for deviation.

By employing these decomposition strategies, we can solve much larger stochastic programming problems than would be possible with a direct deterministic equivalent approach. The "use" comes from enabling the practical application of stochastic programming in real-world scenarios where uncertainty and complex dependencies over time are prevalent.

## 2) Application scenario

Consider a supply chain management problem for a company selling a product with uncertain demand.

*   **Stages:** The problem spans multiple time periods (e.g., months).
*   **Decisions:**  In each period, the company needs to decide how much to order, how much to produce, and how much to ship to various distribution centers.
*   **Uncertainty:**  The demand at each distribution center is uncertain and follows a probability distribution. We can discretize the demand into a set of possible demand scenarios.
*   **Size Blow-up:**  If we have, say, 5 distribution centers, each with 3 possible demand scenarios per month, and we are planning over 12 months, the number of scenarios at the end is potentially 3<sup>(5*12)</sup> which is astronomically large. Even with fewer scenarios, the size grows rapidly.  The deterministic equivalent model would have a huge number of variables representing the decisions in each period under each scenario.  It would be computationally impossible to solve directly.

**Decomposition Solution:** We can use Benders decomposition or SDDP.

*   Benders: The master problem would determine the first-stage decisions (e.g., initial production plan). The subproblems would represent the supply chain operations in subsequent periods under each demand scenario.  The subproblems would generate cuts that are added to the master problem, refining the first-stage decisions to minimize expected costs while accounting for the uncertainty in demand.
*   SDDP: Solves the problem backward in time.  At each stage, it approximates the cost-to-go function (future expected costs) based on sampling scenarios and using cutting plane methods.

## 3) Python method (if possible)

While implementing decomposition methods from scratch can be complex, several Python libraries provide tools and interfaces to work with stochastic programming and implement decomposition algorithms. Examples:

*   **Pyomo:**  A powerful algebraic modeling language in Python that can be used to formulate stochastic programming models.  While it doesn't have built-in decomposition solvers *directly*, it provides the framework to *implement* decomposition algorithms like Benders decomposition yourself by leveraging its modeling capabilities and solver interfaces.

*   **Scikit-Stoch:** A Python package specifically designed for stochastic optimization, although its development seems to have stalled. It provides implementations for some decomposition algorithms.

Here's a simple example using Pyomo to *illustrate* how you *might start* to structure a stochastic program and indicate where decomposition might be applied. Note this *does not* implement the decomposition *itself*; it's a basic model formulation to *demonstrate the setup*:

```python
from pyomo.environ import *

# Model parameters (Simplified)
num_scenarios = 3  # Discretized demand scenarios
num_periods = 2     # Two time periods
demand = {
    (1,1): 10,  # Period 1, Scenario 1: Demand 10
    (1,2): 12,
    (1,3): 15,
    (2,1): 15,  # Period 2, Scenario 1: Demand 15
    (2,2): 18,
    (2,3): 20
}
production_cost = 2
holding_cost = 1

# Create a concrete Pyomo model
model = ConcreteModel()

# Sets
model.PERIODS = RangeSet(num_periods)
model.SCENARIOS = RangeSet(num_scenarios)

# Variables
model.production = Var(model.PERIODS, within=NonNegativeReals)
model.inventory = Var(model.PERIODS, model.SCENARIOS, within=NonNegativeReals)
model.demand_met = Var(model.PERIODS, model.SCENARIOS, within=NonNegativeReals) #Amount of demand met

# Objective Function
def objective_rule(model):
    # Assumes scenarios are equally likely
    scenario_probability = 1.0 / num_scenarios
    return sum(scenario_probability * (
               sum(production_cost * model.production[t] + holding_cost * model.inventory[t, s]
                   for t in model.PERIODS)
            ) for s in model.SCENARIOS)

model.objective = Objective(rule=objective_rule, sense=minimize)


# Constraints

# Inventory balance
def inventory_balance_rule(model, t, s):
    if t == 1:
        return model.inventory[t,s] == model.production[t] - demand[t,s]
    else:
        return model.inventory[t,s] == model.inventory[t-1,s] + model.production[t] - demand[t,s]
model.inventory_balance = Constraint(model.PERIODS, model.SCENARIOS, rule=inventory_balance_rule)

def demand_met_rule(model, t, s):
    return model.demand_met[t,s] <= demand[t,s]

model.demand_met_con = Constraint(model.PERIODS, model.SCENARIOS, rule=demand_met_rule)

def production_positivity_rule(model,t):
    return model.production[t] >= 0
model.production_positive = Constraint(model.PERIODS,rule = production_positivity_rule)
# Solve the model (using a solver like GLPK or CPLEX or Gurobi)
solver = SolverFactory('glpk')  # Or 'cplex', 'gurobi' (if installed)
solver.solve(model)

# Decomposition strategy would involve:
# 1. Identifying stages (e.g., time periods) and/or scenarios that can be separated.
# 2. Formulating a master problem (e.g., for first-stage decisions).
# 3. Formulating subproblems for each scenario or stage.
# 4. Iteratively solving the master problem and subproblems, exchanging information (e.g., cuts).
```

**Explanation:**

*   The Pyomo model sets up a basic inventory management problem under demand uncertainty.
*   The `model.SCENARIOS` set represents the discretized demand scenarios.
*   The `model.objective` calculates the expected cost across all scenarios.
*   The `model.inventory_balance` constraint links decisions and inventory levels across periods for each scenario.

*Important Note:* This code only creates the *deterministic equivalent model*. To *implement* a decomposition strategy (like Benders), you would need to add code to:

1.  Separate the problem into a master problem and subproblems.
2.  Solve the subproblems.
3.  Generate cuts (linear constraints) from the subproblem solutions.
4.  Add the cuts to the master problem.
5.  Iterate until convergence.
6.   Handle the "non-anticipativity constraint" in general which requires first-stage decisions to be equal across all scenarios.

Implementing the decomposition logic within Pyomo requires significant programming effort.

## 4) Follow-up question

Beyond Benders, SDDP, and Progressive Hedging, are there other important decomposition or approximation techniques used in stochastic programming to address the size blow-up issue, and what are their key strengths and weaknesses? For example, what about Sample Average Approximation (SAA) or scenario aggregation techniques? How do they compare to the methods discussed?
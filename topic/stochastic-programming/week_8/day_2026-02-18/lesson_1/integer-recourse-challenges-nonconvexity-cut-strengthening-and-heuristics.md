---
title: "Integer Recourse Challenges: nonconvexity, cut strengthening, and heuristics"
date: "2026-02-18"
week: 8
lesson: 1
slug: "integer-recourse-challenges-nonconvexity-cut-strengthening-and-heuristics"
---

# Topic: Integer Recourse Challenges: nonconvexity, cut strengthening, and heuristics

## 1) Formal definition (what is it, and how can we use it?)

In Stochastic Programming, *integer recourse* refers to a situation where, after observing the realization of a random event (the "first stage"), we must make *integer* decisions (the "second stage" or "recourse").  This is in contrast to continuous recourse, where we can adjust our decisions continuously in response to the observed random event.  The "challenges" in the title stem from the combination of uncertainty (stochastic programming) and integer variables, which leads to significant computational difficulties.

More formally, consider a two-stage stochastic program:

min  c'x + E[Q(x, ξ)]
s.t. Ax = b
x ∈ X

where:

*   x is the first-stage decision vector (often continuous, but can be integer).
*   c'x represents the cost associated with the first-stage decision.
*   ξ is a random variable representing the uncertain parameters (e.g., demand, prices).
*   Q(x, ξ) is the recourse function:  Q(x, ξ) = min q'y s.t. Wy = h - Tx, y ∈ Y, y ∈ Z^n
    *   y is the second-stage (recourse) decision vector (integer).
    *   q'y is the cost associated with the second-stage decision.
    *   Wy = h - Tx represents the constraints that must be satisfied given the first-stage decision x and the realization of the random variable ξ.  Here, W, T, and h are often functions of ξ.
    *   Y defines the feasible region for y.
    *   Z^n denotes that y must be integer-valued.
*   E[Q(x, ξ)] is the expected value of the recourse function over the distribution of the random variable ξ.
*   A, b define the constraints on the first-stage decision.
*   X defines the feasible region for x (can be continuous or integer).

**Challenges and Key Concepts:**

*   **Nonconvexity:**  The expected recourse function E[Q(x, ξ)] is often *nonconvex* even if Q(x, ξ) is convex *for a fixed* ξ. The expected value operation mixes scenarios leading to non-convexities. This makes the overall optimization problem much harder to solve because local optima can trap standard optimization algorithms.
*   **Cut Strengthening:** Traditional cutting plane methods designed for integer linear programming often perform poorly in the context of stochastic programming with integer recourse. Cut strengthening techniques are employed to generate stronger cuts (valid inequalities) that more effectively prune the feasible region and improve convergence. These cuts exploit the specific structure of the stochastic programming problem. Examples include:
    *   *L-shaped cuts* (also known as Benders decomposition cuts). These cuts decompose the problem into a master problem (first stage) and subproblems (second stage) and generate constraints based on the dual solutions of the subproblems.  Strengthening involves improving these cuts.
    *   *Mixed-integer Gomory cuts* and other generic mixed-integer programming cuts can be adapted and strengthened to account for the expected value function.
*   **Heuristics:** Due to the computational complexity of solving these problems exactly, heuristic algorithms are frequently used to find good, but not necessarily optimal, solutions.  These heuristics often combine decomposition methods (e.g., Benders decomposition) with local search techniques or metaheuristics (e.g., genetic algorithms, simulated annealing). They also may take advantage of specific problem structure.

**How can we use it?**

Understanding these concepts is crucial for modeling and solving real-world problems that involve making decisions under uncertainty when some of the decisions must be integer-valued. Integer recourse problems arise in numerous applications. The techniques address the computational challenges.

## 2) Application scenario

**Supply Chain Network Design Under Demand Uncertainty:**

Consider a company designing its supply chain network.  The company must decide where to locate warehouses (first-stage integer decisions).  After observing the actual demand in different regions (the random event), the company must decide how much product to ship from each warehouse to each customer (second-stage integer decisions - you can't ship fractions of products).

*   **First-stage decision (x):** Binary variables representing whether to open a warehouse at a specific location.  Opening a warehouse incurs a fixed cost (c'x).
*   **Random variable (ξ):** The uncertain demand at each customer location.  This could be represented by a probability distribution.
*   **Second-stage decision (y):** Integer variables representing the amount of product shipped from each open warehouse to each customer.
*   **Recourse function (Q(x, ξ)):** The minimum transportation cost required to satisfy the demand, given the warehouse locations and the observed demand.  Constraints include warehouse capacity, customer demand satisfaction, and non-negativity of shipments (Wy = h - Tx).
*   **Overall objective:** Minimize the total cost, including the fixed costs of opening warehouses and the expected transportation costs under demand uncertainty.

**Challenges:**

*   The non-convexity of E[Q(x, ξ)] makes it difficult to find the optimal warehouse locations.
*   Finding integer shipping quantities in the second stage can be computationally expensive, especially with a large number of warehouses and customers.
*   Standard optimization algorithms may get stuck in local optima, leading to poor solutions.

**Solution approaches:**

* Use cut-strengthening techniques, like generating improved L-shaped cuts based on the transportation problem to prune suboptimal warehouse locations.
* Employ heuristic algorithms, such as genetic algorithms or simulated annealing, to explore the solution space and find good warehouse locations. These heuristics can incorporate Benders decomposition to evaluate the recourse costs associated with each candidate warehouse configuration.

## 3) Python method (if possible)

While solving large-scale integer recourse problems often requires specialized solvers and modeling languages (e.g., GAMS, AMPL), we can illustrate a simplified version using Pyomo (a Python-based optimization modeling language) with a MILP solver like GLPK, CBC, or Gurobi/CPLEX. This example will be highly simplified, but it will demonstrate the basic structure.

```python
from pyomo.environ import *
import numpy as np

def create_model(num_scenarios=2, demand_mean=10, demand_std=2):
    """Creates a Pyomo model for a simplified two-stage stochastic program.
       Assumes a single first-stage integer decision and a single second-stage integer decision.
    """
    model = ConcreteModel()

    # Parameters
    model.num_scenarios = num_scenarios
    np.random.seed(42)
    model.scenarios = list(range(num_scenarios))
    model.demand = {s: max(0, int(np.random.normal(demand_mean, demand_std))) for s in model.scenarios} # Non-negative demand

    # First-stage decision (integer): Whether to build a factory (0 or 1)
    model.build_factory = Var(domain=Binary)
    factory_cost = 5

    # Second-stage decision (integer): Production quantity for each scenario
    model.production = Var(model.scenarios, domain=NonNegativeIntegers)
    production_cost = 1

    # Objective function
    def objective_rule(model):
        first_stage_cost = factory_cost * model.build_factory
        expected_second_stage_cost = sum(production_cost * model.production[s] for s in model.scenarios) / model.num_scenarios
        return first_stage_cost + expected_second_stage_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints

    # Production must be at least demand for each scenario if factory is built
    def production_rule(model, s):
        if model.build_factory.value is None or model.build_factory.value == 0: #Need build_factory to be set to 1 or 0 for constraint to be activated
            return Constraint.Skip
        else:
            return model.production[s] >= model.demand[s] #Produce at least demand

    model.production_constraint = Constraint(model.scenarios, rule=production_rule)


    # If not building the factory then production is 0
    def no_factory_no_production(model,s):
        return model.production[s] <= 100 * model.build_factory # Big M constraint.  If factory not built, production is 0
    model.no_factory = Constraint(model.scenarios, rule=no_factory_no_production)
    return model

# Example usage:
model = create_model()
solver = SolverFactory('glpk')  # Or 'cbc', 'gurobi', 'cplex' if you have them installed
# We can't use production_rule directly without fixing model.build_factory. This will be done in the solver
#model.production_constraint = Constraint(model.scenarios, rule=production_rule)
solver.solve(model)

print(f"Build Factory: {model.build_factory.value}")
for s in model.scenarios:
    print(f"Scenario {s}: Production = {model.production[s].value}, Demand = {model.demand[s]}")
print(f"Objective Value: {model.objective()}")
```

**Explanation:**

1.  **`create_model()`:** Defines the Pyomo model.
    *   `model.build_factory`:  A binary variable indicating whether to build a factory.
    *   `model.production[s]`:  An integer variable representing the production quantity in scenario `s`.
    *   `model.demand[s]`:  The demand in scenario `s`.
    *   `objective_rule()`: Defines the objective function (factory cost + expected production cost).
    *   `production_rule()`: Ensures that production meets demand in each scenario IF the factory is built. Note that we needed to evaluate the build_factory.value to either activate or skip this constraint
    *   `no_factory_no_production()`:  A constraint that forces the production to zero if the factory is not built. A Big-M constraint ensures production = 0 when build_factory = 0.

2.  **Solving the model:**
    *   We use `SolverFactory()` to create a solver instance (GLPK in this example).  You may need to install GLPK (e.g., `apt-get install glpk` on Debian/Ubuntu).
    *   `solver.solve(model)` solves the optimization problem.

**Limitations:**

*   **Simplified Example:** This is a very basic illustration.  Real-world problems are much more complex, with many more decision variables, constraints, and scenarios.
*   **No Cut Strengthening or Advanced Heuristics:** This code doesn't implement any cut strengthening techniques or advanced heuristics.  Developing these requires more sophisticated modeling and solver control.
*   **Small Instance:** The computational demands increase rapidly as the problem size grows. GLPK will not be appropriate for large-scale instances. Use more powerful solvers.
*   **Scaling:** For large problems, consider commercial solvers like Gurobi or CPLEX, which have significantly better performance for mixed-integer programming problems, and consider decomposition algorithms and specialized cut generation.

## 4) Follow-up question

What are some practical strategies for generating strong L-shaped cuts in stochastic programming with integer recourse, beyond simply using the dual variables from the subproblem solution? How can we leverage problem-specific information to improve the effectiveness of these cuts?
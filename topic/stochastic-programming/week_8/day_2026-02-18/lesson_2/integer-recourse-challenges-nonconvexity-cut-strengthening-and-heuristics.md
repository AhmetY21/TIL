---
title: "Integer Recourse Challenges: nonconvexity, cut strengthening, and heuristics"
date: "2026-02-18"
week: 8
lesson: 2
slug: "integer-recourse-challenges-nonconvexity-cut-strengthening-and-heuristics"
---

# Topic: Integer Recourse Challenges: nonconvexity, cut strengthening, and heuristics

## 1) Formal definition (what is it, and how can we use it?)

Integer recourse challenges arise in stochastic programming problems where some decision variables in the *recourse* or *second-stage* problem are required to be integers.  Let's break down the components:

*   **Stochastic Programming:** Deals with optimization problems where some parameters are uncertain (random variables).  We often model uncertainty using scenarios, each with an associated probability.

*   **Recourse/Second-Stage Problem:** After observing the realization of the random variables (i.e., which scenario occurred), we can take corrective actions (recourse) to mitigate the impact of the uncertainty. These corrective actions are represented by the second-stage decision variables. The first stage decision is made *before* knowing the realization of the random variable, the second stage decision is made *after*.

*   **Integer Recourse:**  This means that at least some of the second-stage (recourse) decision variables must take integer values (0, 1, 2, ...). This is common in situations where you're deciding on the number of items to produce, the number of facilities to open, the number of vehicles to dispatch, etc.

**Challenges and Nonconvexity:**

The integer constraints in the recourse problem introduce significant challenges compared to continuous recourse. The main issue is **nonconvexity**. When the second-stage problem is an integer program, the resulting objective function (after integrating over all possible scenarios) becomes a piecewise linear, non-convex function.  This makes the overall problem significantly harder to solve.

**Cut Strengthening:**

Cut strengthening techniques aim to improve the linear programming (LP) relaxation of the problem. This is important because many solution algorithms rely on solving LP relaxations. By adding valid inequalities ("cuts") that tighten the LP relaxation without cutting off any integer solutions, we can obtain a better approximation of the feasible region and speed up the solution process.

**Heuristics:**

Due to the complexity of solving integer recourse problems exactly, especially for large instances, heuristic methods are often employed. Heuristics provide good, but not necessarily optimal, solutions within a reasonable amount of time. Examples include:

*   **Relax-and-Fix:**  Iteratively fix subsets of integer variables to their values in the LP relaxation, solve the resulting smaller problem, and repeat.
*   **Local Search:** Starting from a feasible solution, explore the neighborhood of that solution by making small changes (e.g., flipping the value of one integer variable).
*   **Decomposition-based Heuristics:** Decompose the problem into smaller subproblems that are easier to solve and iteratively coordinate the solutions.

**How can we use it?**

Understanding these concepts is crucial for modeling and solving realistic stochastic programming problems where integer decisions are necessary in the recourse phase. By recognizing the challenges, we can choose appropriate solution techniques and potentially develop customized algorithms tailored to the specific problem structure. Cut strengthening helps reduce the integrality gap, while heuristics offer practical solutions for large-scale problems.

## 2) Application scenario

**Supply Chain Disruptions with Facility Location Decisions:**

Imagine a company that needs to decide where to locate distribution centers (DCs) to serve its customers. Demand is uncertain due to potential market fluctuations and supply chain disruptions.  Let's say a natural disaster could damage one or more of the DCs.

*   **First-stage Decision:**  The company must choose which DCs to open *before* knowing whether a disaster will occur. These are binary decisions (0 = don't open, 1 = open).

*   **Uncertainty:**  The location and severity of the natural disaster are uncertain. Each scenario represents a different disaster scenario (e.g., earthquake in region A, flood in region B, no disaster).

*   **Second-stage (Recourse) Decisions:**  *After* a disaster occurs, the company can decide how much to ship from each open DC to each customer to meet demand.  A crucial aspect is that the company may need to **open additional temporary DCs** (e.g., mobile units) to compensate for the damaged facilities. These decisions of *how many* temporary facilities to open at given locations are *integer* recourse variables. These decisions are made *after* we know which scenario happened (the disaster).

**Challenges:**

*   The integer constraint on the number of temporary DCs leads to a nonconvex problem.
*   Solving this problem exactly can be computationally expensive, especially with many potential disaster scenarios and DC locations.

**Solution Approaches:**

*   Cut strengthening could be used to add valid inequalities that exploit the structure of the supply chain network and the integer recourse decisions.
*   A heuristic could be developed that combines a sampling approach (to handle the uncertainty) with a local search algorithm to improve the DC location and shipment decisions.
*   Relax-and-fix could be used by fixing facility locations, solving the recourse subproblem and iterating to find good location assignments.

## 3) Python method (if possible)

While a full solution requires a Mixed Integer Programming (MIP) solver and is too complex to implement here in full, I can demonstrate using `pyomo` how to model a simple two-stage stochastic program with integer recourse and introduce a cut:

```python
import pyomo.environ as pyo
import numpy as np

def create_model(num_scenarios):
    model = pyo.ConcreteModel()

    # Scenario probabilities (uniform for simplicity)
    model.Scenarios = range(num_scenarios)
    model.prob = {s: 1/num_scenarios for s in model.Scenarios}

    # First-stage variable (binary - open/close facility)
    model.x = pyo.Var(within=pyo.Binary)

    # Second-stage variables (integer - units to ship)
    model.y = pyo.Var(model.Scenarios, within=pyo.NonNegativeIntegers) # Integer Recourse

    # Parameters (for simplicity, fixed)
    model.demand = 10
    model.capacity = 5

    # Objective function (minimize cost)
    def obj_rule(model):
        return model.x + sum(model.prob[s] * model.y[s] for s in model.Scenarios)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.MINIMIZE)

    # Constraints

    # First-stage capacity constraint (simplified)
    def cap_rule(model):
        return model.x * model.capacity >= model.demand/2  # Open facility can supply at least half of demand
    model.capacity_con = pyo.Constraint(rule=cap_rule)


    # Second-stage constraints (scenario-dependent)
    def demand_rule(model, s):
        # y[s] represents how many units are met in scenario s
        # This allows for unsatisfied demand
        return model.y[s] <= model.demand
    model.demand_con = pyo.Constraint(model.Scenarios, rule=demand_rule)

    def supply_rule(model, s):
        # In each scenario, the production cannot exceed facility capacity if open,
        #  + any additional supply if the facility is not open
        return model.y[s] <= model.x * model.capacity + (1-model.x) * 10
    model.supply_con = pyo.Constraint(model.Scenarios, rule=supply_rule)

    # **Cut Strengthening Example:**  Add a valid inequality
    # This cut says that on average, the amount supplied in the second stage
    #  should at least cover a portion of the demand
    def cut_rule(model):
        return sum(model.prob[s] * model.y[s] for s in model.Scenarios) >= model.demand * 0.4 # Require 40% average demand coverage
    model.cut = pyo.Constraint(rule=cut_rule)

    return model

# Example Usage:
num_scenarios = 5
model = create_model(num_scenarios)

# Solve the model using a MIP solver (e.g., Gurobi, CPLEX)
solver = pyo.SolverFactory('gurobi')  # Replace with your solver
results = solver.solve(model)

# Print results
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimal Solution Found")
    print(f"First-stage decision (x): {pyo.value(model.x)}")
    for s in model.Scenarios:
        print(f"Second-stage decision (y[{s}]): {pyo.value(model.y[s])}")
    print(f"Objective value: {pyo.value(model.obj)}")
else:
    print("Solver did not find an optimal solution")

```

**Explanation:**

1.  **Model Definition:** The `create_model` function defines the stochastic program using `pyomo`. `model.x` is the first-stage binary decision, and `model.y` are the second-stage *integer* decisions.
2.  **Scenarios:** The model incorporates multiple scenarios, each with an equal probability.
3.  **Objective Function:** The objective is to minimize the sum of the first-stage cost and the expected second-stage cost (using scenario probabilities).
4.  **Constraints:** The model includes capacity constraints for the first-stage decision and scenario-dependent constraints for the second-stage decisions.
5.  **Cut Strengthening:** A cut is added to improve the LP relaxation.  This particular cut requires that, on average across scenarios, the total supply must cover a fraction of the total demand. This is a simplified example of a cut.  More sophisticated cuts can be derived using techniques from integer programming.
6.  **Solving:** The example uses `gurobi` as the solver (you'll need to have it installed and configured correctly). You can replace it with another MIP solver like `cplex`.
7.  **Results:** The code prints the values of the first-stage and second-stage variables, as well as the optimal objective value.

**Important Notes:**

*   This is a *very* simplified example.  Real-world stochastic programs can be much more complex, with many more variables, scenarios, and constraints.
*   The effectiveness of cut strengthening depends heavily on the specific problem structure. Finding good cuts can be challenging and may require domain expertise.
*   Heuristics are often necessary for large-scale instances that cannot be solved to optimality within a reasonable time.

## 4) Follow-up question

How does the choice of scenario generation technique (e.g., Monte Carlo sampling, scenario trees, moment-based approaches) affect the computational complexity and solution quality when dealing with integer recourse problems in stochastic programming? Furthermore, how can we adapt cut generation and heuristic techniques to leverage specific properties of the chosen scenario generation method?
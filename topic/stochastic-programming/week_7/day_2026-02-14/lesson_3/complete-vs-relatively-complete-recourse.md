---
title: "Complete vs Relatively Complete Recourse"
date: "2026-02-14"
week: 7
lesson: 3
slug: "complete-vs-relatively-complete-recourse"
---

# Topic: Complete vs Relatively Complete Recourse

## 1) Formal definition (what is it, and how can we use it?)

**Complete Recourse:**

A stochastic program with recourse is said to have *complete recourse* if a feasible solution to the first-stage decision variables can always be extended to a feasible solution for the second-stage problem, *regardless* of the realization of the random variable (i.e., the scenario).  In other words, no matter what scenario occurs, there is always a way to adjust the second-stage variables to achieve feasibility.

Formally, consider a two-stage stochastic program of the following form:

```
minimize c'x + E[Q(x, ω)]
subject to:
  Ax = b
  x >= 0
```

where:

*   `x` is the first-stage decision vector.
*   `ω` is the random variable.
*   `Q(x, ω)` is the optimal value of the second-stage problem given `x` and `ω`:

```
Q(x, ω) = minimize q(ω)'y
subject to:
  T(ω)x + W(ω)y = h(ω)
  y >= 0
```

*   `y` is the second-stage decision vector.
*   `E[.]` denotes the expectation operator.
*   `A`, `b`, `c`, `T(ω)`, `W(ω)`, `h(ω)`, and `q(ω)` are matrices and vectors of appropriate dimensions, potentially dependent on `ω`.

*Complete recourse* requires that for every feasible `x` satisfying `Ax = b`, `x >= 0`, and for every realization of `ω`, the following set is non-empty:

`{y >= 0 : W(ω)y = h(ω) - T(ω)x}`

This implies that for *every* possible scenario, *any* first-stage decision can be "repaired" by a second-stage decision.  This makes the problem easier to solve because we don't have to explicitly worry about first-stage solutions leading to infeasibility in any particular scenario.

**Relatively Complete Recourse:**

A stochastic program with recourse is said to have *relatively complete recourse* if a feasible solution to the first-stage decision variables can always be extended to a feasible solution for the second-stage problem, *given that* the random variable is within its support. This is a weaker condition than complete recourse. It only guarantees recourse feasibility for realizations that are "possible".

Formally, *relatively complete recourse* requires that for every feasible `x` satisfying `Ax = b`, `x >= 0`, and for every realization of `ω` in the *support* of `ω` (i.e., the set of possible values of `ω` with non-zero probability), the set `{y >= 0 : W(ω)y = h(ω) - T(ω)x}` is non-empty.

**How to Use It:**

*   **Modeling Simplification:**  If a problem naturally possesses complete or relatively complete recourse, you can simplify the model by not explicitly including constraints that enforce feasibility in every scenario.  This can lead to smaller and easier-to-solve problems.
*   **Algorithm Choice:** Some solution algorithms for stochastic programming assume complete or relatively complete recourse. Knowing that the problem has these properties allows you to leverage these specialized algorithms.
*   **Checking for Feasibility:** In problems where recourse is not guaranteed, checking for complete or relatively complete recourse (or its absence) can give insights into the structure of the problem and potential ways to reformulate it to improve solution quality or computational tractability.
*   **Solution Interpretation:** In cases where relatively complete recourse exists, it helps to remember that your solutions will be based on scenarios within the support. Care must be taken in situations that may require decisions outside the modeled range.

## 2) Application scenario

**Complete Recourse Example: Inventory Management**

Imagine a retailer deciding how many units of a product to order in the first stage (the first-stage decision). The demand for the product is uncertain (the random variable).  In the second stage, the retailer can either satisfy the demand or order additional units to meet the demand, paying a penalty for emergency orders. If the retailer can always order enough additional units in the second stage to satisfy demand, regardless of how high the demand turns out to be, then the problem has complete recourse. This is a very common assumption in some simple inventory models.

**Relatively Complete Recourse Example: Power Generation**

Consider a power generation company deciding how much electricity to generate from different sources in the first stage (the first-stage decision).  The demand for electricity is uncertain (the random variable). In the second stage, the company can adjust its power output or purchase electricity from the market. However, the model only considers demand within certain limits.  If the company can always meet the demand within those modeled limits by adjusting its output or purchasing power, then the problem has relatively complete recourse. Outside those limits, it doesn't guarantee feasibility.

## 3) Python method (if possible)

While you cannot directly "test" for complete/relatively complete recourse in a general black box setting *algorithmically*, you can verify it within a specific model formulation using Python and optimization libraries like `Pyomo` or `GurobiPy`.  The general idea is to:

1.  Solve the first-stage problem.
2.  For each scenario (or a representative sample of scenarios):
    *   Fix the first-stage decision variables.
    *   Try to solve the second-stage problem.
    *   If the second-stage problem is feasible for all scenarios, then it *suggests* complete/relatively complete recourse (depending on how scenarios are generated).  If even one scenario is infeasible, then it *proves* that complete/relatively complete recourse does *not* exist.

Here's a simplified example using Pyomo to illustrate the concept. This checks if a recourse action is possible for a limited set of scenarios.  This is just an illustration and doesn't *prove* recourse; a proof requires checking feasibility over the *entire* support of ω, which might be computationally intractable.

```python
from pyomo.environ import *
import numpy as np

def check_recourse(first_stage_solution, scenarios):
    """
    Checks the feasibility of the second-stage problem for a given first-stage
    solution and a set of scenarios.

    Args:
        first_stage_solution (dict): A dictionary mapping first-stage variable names to values.
        scenarios (list of dict): A list of dictionaries, where each dictionary represents
                                 a scenario and contains the values of the random variables.

    Returns:
        bool: True if the second-stage problem is feasible for all scenarios, False otherwise.
    """
    for scenario in scenarios:
        # Create a concrete Pyomo model for the second-stage problem (replace with your actual model)
        model = ConcreteModel()

        # Define second-stage decision variables (replace with your actual variables)
        model.y = Var(within=NonNegativeReals)

        # Define parameters based on the scenario (replace with your actual parameters)
        a_val = scenario['a']  # Value of 'a' in this scenario
        h_val = scenario['h']  # Value of 'h' in this scenario

        # Define the constraint (replace with your actual constraint)
        model.constraint = Constraint(expr= 2*model.y >= h_val - first_stage_solution['x'] - a_val )

        # Define a dummy objective (just to check feasibility)
        model.obj = Objective(expr=model.y, sense=minimize)


        # Solve the second-stage problem
        solver = SolverFactory('glpk')  # Use a free solver like glpk
        results = solver.solve(model)

        # Check the solution status
        if results.solver.termination_condition != TerminationCondition.optimal:
            print(f"Second-stage infeasible for scenario: {scenario}")
            return False #Not feasible

    print("Second-stage feasible for all scenarios checked.")
    return True

# Example Usage
if __name__ == '__main__':

    # First-stage solution (example)
    first_stage_solution = {'x': 2.0}

    # Scenarios (example)
    scenarios = [
        {'a': 1.0, 'h': 3.0},
        {'a': 0.5, 'h': 2.5},
        {'a': 1.5, 'h': 3.5}
    ]

    # Check recourse feasibility
    recourse_feasibility = check_recourse(first_stage_solution, scenarios)

    if recourse_feasibility:
        print("Relatively complete recourse seems plausible based on the scenarios tested.")
    else:
        print("Relatively complete recourse cannot be guaranteed based on the scenarios tested.")

```

**Important Notes:**

*   The code above is a simplified example.  You need to replace the placeholder comments with your actual Pyomo model for the second-stage problem.
*   The choice of scenarios is crucial.  For relatively complete recourse, ensure that the scenarios are representative of the support of the random variable. For complete recourse, you must theoretically check across *all* possible values, which is often impossible in practice, so this usually relies on structural understanding of the model.
*   The code uses `glpk`, a free solver. You might need to install it (e.g., `apt-get install glpk` on Linux).  You can also use other solvers like `Gurobi`, `CPLEX`, etc. if you have licenses for them.
*   This method doesn't *prove* relatively complete recourse in general, but can identify cases where it does not hold and provide empirical support for whether it might hold. You often need analytical techniques to prove complete/relatively complete recourse rigorously.

## 4) Follow-up question

Let's say you are modeling a humanitarian relief effort, and you need to decide where to preposition supplies before a hurricane season. One consideration is that the roads might be blocked or damaged after the hurricane.

*   Would you prioritize a model with complete recourse or relatively complete recourse in this case? Why? What are the potential drawbacks of each?
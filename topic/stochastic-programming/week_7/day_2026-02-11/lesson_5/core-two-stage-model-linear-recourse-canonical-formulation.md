---
title: "Core Two-Stage Model (Linear Recourse) — canonical formulation"
date: "2026-02-11"
week: 7
lesson: 5
slug: "core-two-stage-model-linear-recourse-canonical-formulation"
---

# Topic: Core Two-Stage Model (Linear Recourse) — canonical formulation

## 1) Formal definition (what is it, and how can we use it?)

The *core two-stage stochastic programming model with linear recourse* is a fundamental model for decision-making under uncertainty. It represents situations where we make a decision in the first stage *before* the uncertainty is realized, and then in the second stage, we take *recourse* actions to mitigate the impact of the realized uncertainty. The term "linear recourse" refers to the fact that the second-stage problem is a linear program. The canonical formulation highlights the structure of the model for analysis and optimization.

**Formal Definition:**

The core two-stage stochastic program with linear recourse can be written as:

```
minimize  c'x + E[Q(x, ξ)]
subject to Ax = b
            x >= 0
```

where:

*   **x** is the first-stage decision vector.  This represents the decisions we need to make *before* knowing the outcome of the random event.
*   **c** is the cost vector associated with the first-stage decisions.  `c'x` (c transpose x) represents the cost of the first-stage decision.
*   **ξ** is the random vector (representing the uncertainty) with a known probability distribution.
*   **E[Q(x, ξ)]** is the expected value of the second-stage cost, often referred to as the recourse function.
*   **A** and **b** are the coefficient matrix and right-hand side vector, respectively, of the first-stage constraints.  These constraints must be satisfied by the first-stage decisions.
*   **Q(x, ξ)** is the optimal value of the second-stage problem given the first-stage decision *x* and the realization of the random variable *ξ*.  The second-stage problem is a linear program:

    ```
    Q(x, ξ) = minimize  q'y
             subject to  Ty = h - Wx
                         y >= 0
    ```

    *   **y** is the second-stage decision vector (recourse actions).
    *   **q** is the cost vector associated with the second-stage decisions.  `q'y` represents the cost of the recourse actions.
    *   **T**, **W**, and **h** are matrices and vectors that define the constraints of the second-stage problem.
        *   **T** is the technology matrix for the second-stage problem.
        *   **W** is the recourse matrix, linking the first-stage decisions *x* to the second-stage constraints. `Wx` represents how the first-stage decisions affect the second-stage constraints.
        *   **h** is a vector representing the right-hand side of the second-stage constraints, *before* considering the impact of the first-stage decision and the random outcome.
    *   `h - Wx` is often referred to as the *right-hand side* of the second-stage problem, which depends on both the first-stage decision `x` and the realization of the random variable `ξ` (since `h` contains the random component).

**How can we use it?**

This model can be used to optimize decisions where there are two stages. First, you decide on a course of action (`x`). Then, based on what actually happens (the realization of `ξ`), you take corrective actions (`y`) to minimize costs, considering the initial decision. Crucially, the model allows us to *anticipate* the need for these corrective actions when making the first-stage decisions, thereby improving overall performance. The expectation `E[Q(x, ξ)]` integrates over the possible outcomes of the random variable, allowing the first-stage decision to be made "robustly" against uncertainty.

## 2) Application scenario

Consider a **supply chain planning problem**.

*   **First Stage:** A company needs to decide how much product to manufacture at its factory *before* knowing the exact demand in the upcoming period. `x` represents the production quantity. `c` represents the cost of production.
*   **Uncertainty:** The demand for the product is uncertain. `ξ` represents the random demand.  We might assume a discrete set of scenarios, each with an associated probability, to represent different demand levels.
*   **Second Stage (Recourse):**  After the demand is realized, the company can take one of two actions:
    *   If production exceeds demand, the company can store the excess product.
    *   If demand exceeds production, the company can purchase additional product from a more expensive supplier. `y` represents the amount of product stored (if `y` > 0) or purchased from the expensive supplier (if `y` < 0). `q` represents the cost of storage (if y > 0) or the higher purchase cost (if y < 0).
*   **Constraints:**
    *   First-stage constraints (`Ax = b`) might include capacity constraints on the factory.
    *   Second-stage constraints (`Ty = h - Wx`) would ensure that demand is met.  `h` represents the realized demand (the random component) and `Wx` represents the amount of product available from the factory (the first-stage decision).  `Ty = h - Wx` ensures that the recourse action `y` makes up the difference between production and demand.

The objective is to minimize the expected total cost, which includes the cost of production in the first stage and the expected cost of storage or emergency purchases in the second stage. This framework allows the company to make a production decision that balances the cost of overproduction (storage) and underproduction (expensive emergency purchases), taking into account the probability distribution of demand.

## 3) Python method (if possible)

Here's an example using Pyomo and a simple discrete scenario approach.  Note that solving stochastic programs generally involves approximating the distribution of the random variable with a finite number of scenarios.  There are other approaches, but this illustrates the core concepts.

```python
from pyomo.environ import *
import numpy as np

def create_two_stage_model(scenarios, probabilities, c, A, b, q, T, W, h):
    """
    Creates a Pyomo model for a two-stage stochastic program with linear recourse.

    Args:
        scenarios: List of random variable scenarios (e.g., demand levels).
        probabilities: List of probabilities corresponding to each scenario.
        c: First-stage cost vector.
        A: First-stage constraint matrix.
        b: First-stage constraint right-hand side vector.
        q: Second-stage cost vector.
        T: Second-stage technology matrix.
        W: Second-stage recourse matrix.
        h: Second-stage right-hand side vectors for each scenario.  This should be a list
           of vectors, one for each scenario.

    Returns:
        A Pyomo ConcreteModel object.
    """

    model = ConcreteModel()

    # First-stage variables
    model.x = Var(range(len(c)), within=NonNegativeReals) # Example: production amount
    # First-stage constraints
    model.first_stage_constraints = ConstraintList()
    for i in range(len(b)):
        model.first_stage_constraints.add(sum(A[i, j] * model.x[j] for j in range(len(c))) == b[i])

    # Scenarios
    model.Scenarios = Set(initialize=range(len(scenarios)))

    # Second-stage variables (indexed by scenario)
    model.y = Var(model.Scenarios, range(len(q)), within=NonNegativeReals) # Example: Amount purchased or stored

    # Second-stage constraints (indexed by scenario)
    def second_stage_constraint_rule(model, s, i):
        return sum(T[i, j] * model.y[s, j] for j in range(len(q))) == h[s][i] - sum(W[i, k] * model.x[k] for k in range(len(c)))
    model.second_stage_constraints = Constraint(model.Scenarios, range(h[0].shape[0]), rule=second_stage_constraint_rule)

    # Objective function
    def objective_rule(model):
        first_stage_cost = sum(c[i] * model.x[i] for i in range(len(c)))
        second_stage_cost = sum(probabilities[s] * sum(q[j] * model.y[s, j] for j in range(len(q))) for s in model.Scenarios)
        return first_stage_cost + second_stage_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    return model

# Example usage (replace with your actual data)
if __name__ == '__main__':
    # Problem parameters (example)
    scenarios = [10, 15, 20]  # Possible demand levels
    probabilities = [0.3, 0.4, 0.3] # Probabilities of each demand level

    c = np.array([1.0])  # Cost per unit of production
    A = np.array([[1.0]])  # Capacity constraint example (x <= 25)
    b = np.array([25.0])

    q = np.array([2.0])  # Cost per unit purchased (if demand exceeds production)
    T = np.array([[1.0]])  # Second-stage technology matrix
    W = np.array([[1.0]])  # Recourse matrix - production offsets demand
    h = [np.array([s]) for s in scenarios] # RHS for each scenario (demand level)

    # Create the model
    model = create_two_stage_model(scenarios, probabilities, c, A, b, q, T, W, h)

    # Solve the model
    solver = SolverFactory('glpk')  # or 'ipopt', etc.  Install first if needed!
    solver.solve(model)

    # Print the results
    print("First-stage decision (production):", value(model.x[0]))
    for s in model.Scenarios:
        print(f"Scenario {s}: Demand = {scenarios[s]}, Recourse action (purchase) = {value(model.y[s, 0])}")
    print("Objective function value:", value(model.objective))
```

**Explanation of the code:**

1.  **`create_two_stage_model` function:** This function takes the problem parameters (scenarios, probabilities, costs, constraints, etc.) as input and constructs the Pyomo model.
2.  **First-stage variables and constraints:**  The `x` variables are defined, representing the first-stage decisions, and the first-stage constraints (`Ax=b`) are implemented using a `ConstraintList`.
3.  **Scenarios:**  The `Scenarios` set represents the possible realizations of the random variable.
4.  **Second-stage variables and constraints:**  The `y` variables are indexed by the scenario and represent the second-stage (recourse) actions.  The second-stage constraints (`Ty = h - Wx`) are implemented using a `Constraint`, indexed by the scenario. `h[s]` is the right hand side vector for scenario `s`.
5.  **Objective function:** The objective function minimizes the sum of the first-stage cost and the *expected* second-stage cost (weighted by the probabilities of each scenario).
6.  **Example Usage:** The `if __name__ == '__main__':` block provides a simple example of how to use the function with some dummy data.  You would replace this with your actual data. It then sets up and solves the Pyomo model using GLPK (or another solver you have installed).  Finally, it prints the results, including the optimal first-stage decision and the optimal recourse actions for each scenario.

**Important notes:**

*   **Solver:** You'll need a suitable solver installed (e.g., GLPK, IPOPT). Install them using `pip install pyomo`. Then install the solver using `apt-get install glpk` or similar depending on your operating system.
*   **Data:** This code provides a *template*.  You will need to replace the example data with your specific problem data.
*   **Scenario Generation:**  The code assumes you have a discrete set of scenarios. Generating realistic scenarios is a critical step in stochastic programming.
*   **Scaling:** For large-scale problems, the scenario-based approach can become computationally expensive.  Consider alternative solution techniques, such as decomposition methods.
*   **Error Handling:** The example does not include extensive error handling. In a real application, you should add checks to ensure the input data is valid and the solver finds a solution.

## 4) Follow-up question

How does the complexity of solving a core two-stage stochastic program with linear recourse scale with the number of scenarios?  What are some techniques to mitigate this issue for large-scale problems?
---
title: "Two-Stage Decomposition: Benders (L-shaped) method — idea"
date: "2026-02-16"
week: 8
lesson: 6
slug: "two-stage-decomposition-benders-l-shaped-method-idea"
---

# Topic: Two-Stage Decomposition: Benders (L-shaped) method — idea

## 1) Formal definition (what is it, and how can we use it?)

The Benders decomposition method, also known as the L-shaped method, is a technique for solving two-stage stochastic linear programming problems (or problems with similar block angular structure). These problems involve making decisions in two stages:

*   **First-stage decisions (here-and-now decisions):**  These decisions must be made before the realization of any uncertainty. They are represented by a vector *x*.

*   **Second-stage decisions (recourse decisions):** These decisions can be made *after* observing the uncertain parameters. They are represented by a vector *y*. The second-stage decisions are designed to "recourse" or correct for any adverse effects of the uncertainty on the first-stage decisions.

The typical two-stage stochastic linear programming problem can be written as:

```
min c'x + E[Q(x, ξ)]
s.t. Ax = b
x ∈ X
```

Where:

*   *x* is the vector of first-stage decision variables.
*   *c* is the cost vector associated with the first-stage variables.
*   *ξ* represents the random parameters (e.g., demand, costs).
*   *E[Q(x, ξ)]* is the expected value of the second-stage cost.
*   *A* is a matrix of coefficients for the first-stage constraints.
*   *b* is the right-hand side vector for the first-stage constraints.
*   *X* is the feasible set for the first-stage variables (e.g., non-negativity constraints).
*   *Q(x, ξ)* is the optimal value of the second-stage problem, given *x* and the realization of *ξ*. The second-stage problem is:

```
Q(x, ξ) = min q'y
s.t. Wy = h - Tx
y ∈ Y
```

Where:

*   *y* is the vector of second-stage decision variables.
*   *q* is the cost vector associated with the second-stage variables.
*   *W* is the matrix of coefficients for the second-stage constraints.
*   *h* is the right-hand side vector for the second-stage constraints (dependent on *ξ*).
*   *T* is a matrix relating the first-stage variables to the second-stage constraints.
*   *Y* is the feasible set for the second-stage variables (e.g., non-negativity constraints).

**Idea of the L-shaped method:**

The L-shaped method decomposes the problem into two parts:

1.  **Master Problem:** Deals with the first-stage variables *x* and approximates the expected second-stage cost *E[Q(x, ξ)]* using linear cuts.

2.  **Subproblems:** For a given *x*, these are the second-stage problems solved for different realizations of the random parameters *ξ*. These subproblems generate feasibility and optimality cuts that are added to the master problem to improve the approximation of *E[Q(x, ξ)]*.

The algorithm iteratively solves the master problem and subproblems, adding cuts to the master problem based on the subproblem solutions. This process continues until a satisfactory solution is found. The cuts are of two types:

*   **Optimality Cuts:**  These cuts approximate the cost function E[Q(x, ξ)] from below.
*   **Feasibility Cuts:**  These cuts eliminate first-stage solutions *x* that lead to infeasible second-stage problems for some realization of *ξ*.

The L-shaped method allows us to solve large-scale stochastic programs by iteratively improving the first-stage solution and progressively refining the approximation of the second-stage cost.

## 2) Application scenario

A common application is **supply chain management under demand uncertainty**.

Imagine a company that needs to decide how much inventory to stock in its warehouses (*x* - first-stage decision). The demand for the product is uncertain (*ξ* - random parameter). After observing the actual demand, the company can decide how much to produce or purchase to meet the demand (*y* - second-stage decision).

*   **Objective:** Minimize the total cost, including inventory holding costs, production/purchase costs, and potential shortage costs.
*   **First-Stage:** Deciding on the initial inventory levels at warehouses before knowing the true demand.
*   **Second-Stage:** Determining the optimal production/purchase quantities after observing the actual demand to satisfy customer needs, potentially incurring penalties for shortages.

The Benders decomposition method can be used to find the optimal inventory levels (*x*) by iteratively solving a master problem that minimizes the first-stage costs plus an approximation of the expected second-stage costs (production/purchase and shortage costs). The subproblems would involve solving the second-stage problem for different demand scenarios to generate cuts that refine the approximation in the master problem.

Another scenario is **investment planning with uncertain returns**.  You must decide on the amount of capital to invest in different projects (x) now.  The returns of the projects depend on uncertain market conditions (ξ).  In the second stage, you can re-allocate resources or invest in different projects (y) to optimize your overall profit.

## 3) Python method (if possible)
```python
import pyomo.environ as pyo
import numpy as np

def benders_decomposition(c, A, b, q, W, T, h, num_scenarios):
    """
    Implements a simplified Benders decomposition for a two-stage stochastic LP.

    Args:
        c: Cost vector for first-stage variables (x).
        A: Constraint matrix for first-stage constraints.
        b: Right-hand side vector for first-stage constraints.
        q: Cost vector for second-stage variables (y).
        W: Constraint matrix for second-stage constraints.
        T: Matrix relating first-stage variables to second-stage constraints.
        h: Right-hand side vector for second-stage constraints (scenarios).
        num_scenarios: Number of scenarios to consider.

    Returns:
        x_optimal: Optimal first-stage solution.
    """

    # Master Problem
    master_model = pyo.ConcreteModel()
    master_model.x = pyo.Var(range(len(c)), within=pyo.NonNegativeReals)  # First-stage variables
    master_model.theta = pyo.Var(within=pyo.Reals)  # Approximation of expected second-stage cost

    # Objective Function
    master_model.objective = pyo.Objective(expr=sum(c[i] * master_model.x[i] for i in range(len(c))) + master_model.theta,
                                        sense=pyo.minimize)

    # First-stage Constraints
    master_model.first_stage_constraints = pyo.ConstraintList()
    for i in range(len(b)):
        master_model.first_stage_constraints.add(expr=sum(A[i, j] * master_model.x[j] for j in range(len(c))) <= b[i])

    # Optimality Cuts (Initially empty)
    master_model.optimality_cuts = pyo.ConstraintList()

    # Optimizer
    solver = pyo.SolverFactory('glpk')  # You can use other solvers like 'gurobi' or 'cplex'

    # Benders Iteration
    max_iterations = 10
    for iteration in range(max_iterations):
        print(f"Iteration: {iteration + 1}")

        # Solve Master Problem
        solver.solve(master_model)
        x_val = [pyo.value(master_model.x[i]) for i in range(len(c))]  # Current first-stage solution
        theta_val = pyo.value(master_model.theta)

        # Subproblem (Second-Stage Problem)
        subproblem_model = pyo.ConcreteModel()
        subproblem_model.y = pyo.Var(range(len(q)), within=pyo.NonNegativeReals)  # Second-stage variables
        subproblem_model.scenario = pyo.Param(initialize=0) # parameter used to select the correct h.

        # Objective Function
        subproblem_model.objective = pyo.Objective(expr=sum(q[i] * subproblem_model.y[i] for i in range(len(q))),
                                                sense=pyo.minimize)

        # Second-stage Constraints
        subproblem_model.constraints = pyo.ConstraintList()
        for i in range(W.shape[0]):
            subproblem_model.constraints.add(expr=sum(W[i, j] * subproblem_model.y[j] for j in range(len(q))) == h[subproblem_model.scenario.value, i] - sum(T[i,j] * x_val[j] for j in range(len(c))))


        # Evaluate subproblem for each scenario and generate cuts
        expected_cost = 0
        for scenario in range(num_scenarios):
            subproblem_model.scenario.value = scenario
            subproblem_instance = subproblem_model.create_instance()
            results = solver.solve(subproblem_instance)

            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                print("Subproblem infeasible or unbounded. This implementation requires modifications to handle this.")
                return None # need to add feasibility cuts to the master problem when this is needed

            obj_val = pyo.value(subproblem_instance.objective)
            expected_cost += obj_val

        expected_cost /= num_scenarios # average over all scenarios

        #Generate Optimality Cut (assuming dual solutions are available from the subproblems - simplified)
        master_model.optimality_cuts.add(master_model.theta >= expected_cost ) # replace with correct dual value from subproblem if dual values are available.

        #Convergence check is not rigorously implemented. This is a simplified example.
        if abs(theta_val - expected_cost) < 0.001:
             print("Convergence reached.")
             break

    x_optimal = [pyo.value(master_model.x[i]) for i in range(len(c))]
    return x_optimal


# Example Usage (small illustrative example)
if __name__ == '__main__':
    # First-stage data
    c = np.array([2, 3])  # Cost vector for x
    A = np.array([[1, 1]])  # Constraint matrix for x
    b = np.array([5])  # Right-hand side for x constraints

    # Second-stage data
    q = np.array([1, 2])  # Cost vector for y
    W = np.array([[1, 1]])  # Constraint matrix for y
    T = np.array([[1, 1]])  # Matrix relating x to y constraints
    h = np.array([[7, 8], [6,9]])  # Right-hand side for y constraints (scenarios)
    num_scenarios = 2

    # Run Benders Decomposition
    x_optimal = benders_decomposition(c, A, b, q, W, T, h, num_scenarios)

    if x_optimal:
        print("Optimal first-stage solution:", x_optimal)
    else:
        print("Benders decomposition failed.")
```

**Explanation:**

1.  **`benders_decomposition(c, A, b, q, W, T, h, num_scenarios)` function:**
    *   Takes the problem parameters as input.
    *   Creates a Pyomo model for the master problem.
    *   Iteratively solves the master problem and subproblems.
    *   Adds optimality cuts to the master problem based on the subproblem solutions.
    *   Returns the optimal first-stage solution `x_optimal`.

2.  **Master Problem:**
    *   `master_model.x`: First-stage decision variables.
    *   `master_model.theta`: Approximation of the expected second-stage cost.  The algorithm tries to find the best x, while finding a theta that is a lower bound on the average cost from the subproblem.  As the cuts are added, theta will increase (or stay the same) towards the real expected second stage cost.
    *   `master_model.objective`: Minimizes the first-stage cost plus the approximation of the second-stage cost.
    *   `master_model.first_stage_constraints`: First-stage constraints.
    *   `master_model.optimality_cuts`: List of optimality cuts (initially empty).

3.  **Subproblem:**
    *   `subproblem_model.y`: Second-stage decision variables.
    *   `subproblem_model.objective`: Minimizes the second-stage cost.
    *   `subproblem_model.constraints`: Second-stage constraints, depending on the first-stage solution *x* and the scenario *ξ*.

4.  **Benders Iteration:**
    *   Solves the master problem to get a candidate first-stage solution *x*.
    *   Solves the subproblem for each scenario to evaluate the second-stage cost given *x*.
    *   Calculates the expected second-stage cost across all scenarios.
    *   Generates an optimality cut based on the subproblem solutions. **Important:** This simplified example directly uses the optimal objective value of the subproblem in the cut. A proper implementation would typically use the dual variables of the subproblem to form the cut.
    *   Adds the optimality cut to the master problem.
    * A convergence check is used to see if `theta_val` is close to the `expected_cost`

5.  **Example Usage:**
    *   Provides a small example problem to demonstrate the usage of the `benders_decomposition` function.

**Important Considerations:**

*   **Dual Variables:**  A true Benders implementation relies on the *dual variables* of the second-stage problem to construct the cuts.  The dual variables contain information about the sensitivity of the optimal objective value to changes in the constraints, which is crucial for generating effective cuts.  This simplified example *approximates* this by using the optimal objective value directly.  A real implementation should calculate dual solutions.
*   **Feasibility Cuts:** The provided code *doesn't include feasibility cuts*.  If the subproblem is infeasible for some scenarios, you need to add feasibility cuts to the master problem to exclude first-stage solutions that lead to infeasibility. This requires solving the subproblem to determine the extreme rays of the dual feasible region and using these to formulate the cuts.
*   **Convergence:** The convergence check is very rudimentary. Proper Benders implementations use tighter convergence criteria based on duality gaps and objective function bounds.
*   **Solver:**  You'll need a suitable LP solver installed (e.g., GLPK, Gurobi, CPLEX) and accessible through Pyomo.
*   **Scenario Generation:** For problems with a large number of scenarios, you'll typically use scenario generation techniques (e.g., Monte Carlo simulation, sampling methods) to generate a representative set of scenarios.
*   **Integer Variables:**  This code focuses on linear programming.  For problems with integer variables, the Benders decomposition algorithm needs to be adapted, typically using branch-and-cut techniques.
*   **Scalability:** Benders decomposition can be very effective for certain problem structures, but its performance depends heavily on the formulation and the properties of the subproblems.

This simplified example provides a basic understanding of the Benders decomposition idea.  A complete and robust implementation would require handling dual variables, feasibility cuts, and more sophisticated convergence criteria.

## 4) Follow-up question

How do you handle infeasibility in the second-stage problem within the Benders decomposition framework, and what form do the corresponding "feasibility cuts" take?
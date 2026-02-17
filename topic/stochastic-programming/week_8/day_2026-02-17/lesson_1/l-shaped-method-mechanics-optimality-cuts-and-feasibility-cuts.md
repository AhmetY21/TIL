---
title: "L-shaped Method Mechanics: optimality cuts and feasibility cuts"
date: "2026-02-17"
week: 8
lesson: 1
slug: "l-shaped-method-mechanics-optimality-cuts-and-feasibility-cuts"
---

# Topic: L-shaped Method Mechanics: optimality cuts and feasibility cuts

## 1) Formal definition (what is it, and how can we use it?)

The L-shaped method is a decomposition algorithm specifically designed to solve two-stage stochastic linear programming problems with recourse. These problems have the general form:

`min  c'x + Q(x)`
`s.t. Ax = b`
`     x >= 0`

where:

*   `x` is the first-stage decision vector (chosen before the uncertainty is revealed).
*   `c` is the cost vector for the first-stage decision.
*   `A` and `b` define the constraints for the first-stage decision.
*   `Q(x)` is the *recourse function*, representing the expected cost of the optimal second-stage decisions taken *after* the uncertainty is realized and the first-stage decision `x` is fixed.  The key is that calculating `Q(x)` is typically computationally expensive.

The core idea of the L-shaped method is to approximate `Q(x)` with a set of linear inequalities (cuts) and iteratively refine this approximation.  The method involves solving a master problem and a set of subproblems, generating *optimality cuts* and *feasibility cuts* based on the solutions.

**Optimality Cuts (Benders Cuts):**  These cuts provide a lower bound approximation to the recourse function `Q(x)`. They are derived from the dual solutions of the second-stage subproblems.  If, for a given `x`, the second-stage problem is feasible and bounded, we can define an optimality cut of the form:

`θ >= α + β'x`

where:

*   `θ` represents an approximation of `Q(x)` (an auxiliary variable added to the master problem).
*   `α` and `β` are coefficients derived from the dual solutions of the second-stage subproblems. Specifically, `α = Σ_ω p_ω π_ω'T_ω` and `β = Σ_ω p_ω π_ω'W_ω`. Here, `ω` indexes the scenarios, `p_ω` are the scenario probabilities, `π_ω` are the dual variables corresponding to the second-stage constraints, and `T_ω` and `W_ω` are matrices from the second-stage formulation described below.

**Feasibility Cuts:**  These cuts ensure that the first-stage solution `x` leads to a feasible second-stage problem for all scenarios.  If, for a given `x`, a second-stage problem is *infeasible*, we generate a feasibility cut of the form:

`0 >= γ + δ'x`

where `γ` and `δ` are coefficients derived from the dual solutions of the infeasibility form of the second-stage subproblems. Specifically, `γ = Σ_ω p_ω v_ω'h_ω` and `δ = Σ_ω p_ω v_ω'T_ω`.  Here, `v_ω` are the dual variables related to infeasibility, and `h_ω` and `T_ω` are relevant matrices from the second-stage formulation.  These cuts effectively exclude first-stage decisions `x` that would lead to infeasible second-stage problems under any scenario.

**Second-Stage Problem Formulation (for context):**

For each scenario `ω`, the second-stage problem is:

`min q_ω'y_ω`
`s.t. W_ω y_ω = h_ω - T_ω x`
`     y_ω >= 0`

where:

* `y_ω` is the second-stage decision variable.
* `q_ω` is the cost vector for the second-stage decision.
* `W_ω`, `T_ω`, and `h_ω` are matrices and vectors defining the second-stage constraints.

**Usage:**

The L-shaped method is used to find the optimal first-stage decision `x` while considering the expected costs and feasibility of the second-stage decisions under various scenarios. It iteratively improves the approximation of the recourse function until a stopping criterion is met (e.g., a small gap between the lower and upper bounds on the optimal objective value).

## 2) Application scenario

Consider a supply chain network design problem. A company needs to decide where to build distribution centers (first-stage decision `x`). The demand in different regions is uncertain (different scenarios).  After building the distribution centers, the company must decide how much product to ship from each distribution center to each customer to meet the demand (second-stage decision `y`).

*   `x`: Binary variables indicating whether to build a distribution center at a specific location.
*   `c`: Cost of building a distribution center.
*   `y`: Quantity of product shipped from each distribution center to each customer.
*   `q`: Transportation cost per unit shipped.
*   The uncertainty lies in the customer demand for each region. Scenarios represent different demand levels.

The L-shaped method can be used to determine the optimal locations for the distribution centers, considering the expected transportation costs under different demand scenarios and ensuring that the company can meet the demand in all scenarios.  Feasibility cuts would prevent the algorithm from selecting distribution center locations that cannot satisfy demand in some scenarios.  Optimality cuts would refine the estimated cost of the second-stage recourse based on the chosen distribution center locations.

## 3) Python method (if possible)

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def l_shaped_method(c, A, b, W, T, h, q, scenarios, probabilities, epsilon=1e-6, max_iterations=100):
    """
    Implements the L-shaped method for two-stage stochastic linear programming.

    Args:
        c: First-stage cost vector.
        A: First-stage constraint matrix.
        b: First-stage constraint vector.
        W: A list of second-stage constraint matrices (one for each scenario).
        T: A list of matrices linking first-stage and second-stage decisions (one for each scenario).
        h: A list of second-stage right-hand-side vectors (one for each scenario).
        q: A list of second-stage cost vectors (one for each scenario).
        scenarios: Number of scenarios.
        probabilities: A list of scenario probabilities.
        epsilon: Optimality tolerance.
        max_iterations: Maximum number of iterations.

    Returns:
        x_optimal: Optimal first-stage decision vector.
        theta_optimal: Optimal value of the recourse function approximation.
        lower_bound: Lower bound on the optimal objective value.
        upper_bound: Best upper bound seen.
        iteration: Number of iterations performed.
    """
    n_x = len(c)  # Number of first-stage variables

    # Initialize master problem
    master_model = gp.Model("MasterProblem")
    x = master_model.addVars(n_x, vtype=GRB.CONTINUOUS, name="x")  # Changed from BINARY for demonstration
    theta = master_model.addVar(vtype=GRB.CONTINUOUS, name="theta") # Auxiliary variable
    master_model.setObjective(sum(c[i] * x[i] for i in range(n_x)) + theta, GRB.MINIMIZE)

    # First-stage constraints
    for i in range(len(b)):
        master_model.addConstr(sum(A[i][j] * x[j] for j in range(n_x)) == b[i], name=f"first_stage_{i}")

    # Non-negativity constraints
    for i in range(n_x):
        master_model.addConstr(x[i] >= 0, name=f"x_nonneg_{i}")

    optimality_cuts = []
    feasibility_cuts = []

    lower_bound = -float('inf')
    upper_bound = float('inf')

    for iteration in range(max_iterations):
        master_model.optimize()

        if master_model.status != GRB.OPTIMAL:
            print("Master problem infeasible or unbounded.  Check data.")
            return None, None, None, None, iteration

        x_val = [x[i].x for i in range(n_x)]
        theta_val = theta.x
        current_objective = master_model.objVal

        Q_x = 0.0 # Expected second-stage cost

        # Solve subproblems for each scenario
        infeasible_scenario_found = False
        for omega in range(scenarios):
            subproblem = gp.Model(f"Subproblem_Scenario_{omega}")
            y = subproblem.addVars(len(q[omega]), vtype=GRB.CONTINUOUS, name="y")

            subproblem.setObjective(sum(q[omega][j] * y[j] for j in range(len(q[omega]))), GRB.MINIMIZE)

            # Recourse constraints
            rhs = h[omega] - np.dot(T[omega], x_val)

            for i in range(len(rhs)):
                subproblem.addConstr(sum(W[omega][i][j] * y[j] for j in range(len(q[omega]))) == rhs[i], name=f"recourse_{i}")

            for i in range(len(q[omega])):
                subproblem.addConstr(y[i] >= 0, name=f"y_nonneg_{i}")
            subproblem.Params.OutputFlag = 0 # Suppress output

            subproblem.optimize()

            if subproblem.status == GRB.OPTIMAL:
                Q_x += probabilities[omega] * subproblem.objVal
            elif subproblem.status == GRB.INFEASIBLE:
                # Generate feasibility cut
                print(f"Infeasible subproblem in scenario {omega}, generating feasibility cut.")
                infeasible_scenario_found = True
                subproblem.Params.OutputFlag = 1
                subproblem.computeIIS() # Irreducible Inconsistent Subsystem
                infeas_constr = None
                for constr in subproblem.getConstrs():
                  if constr.IISConstr:
                    infeas_constr = constr
                    break
                # Dualize the infeasible subproblem to derive the feasibility cut
                # (This part would involve formulating and solving the dual, which is omitted for brevity but is crucial)
                # v_val = dual variables for constraints  W_omega y_omega = h_omega - T_omega x_val
                # Assume the dual values are available: v_val = ...
                # γ = sum(probabilities[omega] * v_val[i] * h[omega][i])
                # δ = sum(probabilities[omega] * sum(v_val[i] * T[omega][i][j] for j in range(n_x)) )

                # For demonstration purposes, create a dummy feasibility cut. In reality, calculate gamma and delta.
                gamma = 1.0  # Example value, needs to be calculated
                delta = np.ones(n_x)  # Example values, needs to be calculated

                feasibility_cut = master_model.addConstr(sum(delta[i] * x[i] for i in range(n_x)) <= -gamma, name="feasibility_cut")
                feasibility_cuts.append(feasibility_cut)

                master_model.remove(theta) # Remove the recourse approximation variable
                master_model = gp.Model("MasterProblem")
                x = master_model.addVars(n_x, vtype=GRB.CONTINUOUS, name="x")
                # Add all cuts to the Master Problem
                for cut in feasibility_cuts:
                    master_model.addConstr(cut)
                master_model.setObjective(sum(c[i] * x[i] for i in range(n_x)), GRB.MINIMIZE)

                # First-stage constraints
                for i in range(len(b)):
                    master_model.addConstr(sum(A[i][j] * x[j] for j in range(n_x)) == b[i], name=f"first_stage_{i}")

                # Non-negativity constraints
                for i in range(n_x):
                    master_model.addConstr(x[i] >= 0, name=f"x_nonneg_{i}")
                break  # No need to solve other subproblems if one is infeasible

        if infeasible_scenario_found:
          print("Skipping optimality cut generation since a feasibility cut was generated")
          continue # Move to the next iteration

        # Generate optimality cut
        if abs(Q_x - theta_val) > epsilon:
            print(f"Generating optimality cut, Q_x = {Q_x}, theta = {theta_val}")
            # Dualize the feasible subproblems to derive the optimality cut
            # (This part would involve formulating and solving the dual, which is omitted for brevity but is crucial)
            # pi_val = dual variables for constraints W_omega y_omega = h_omega - T_omega x_val

            # Assume the dual values are available: pi_val = ...
            # alpha = sum(probabilities[omega] * sum(pi_val[i] * h[omega][i]))
            # beta = sum(probabilities[omega] * sum(pi_val[i] * T[omega][i][j] for j in range(n_x)) )

            # For demonstration purposes, create a dummy optimality cut. In reality, calculate alpha and beta.
            alpha = Q_x # Expected Value
            beta = np.zeros(n_x) # Dummy value for x
            optimality_cut = master_model.addConstr(theta >= alpha + sum(beta[i] * x[i] for i in range(n_x)), name="optimality_cut")
            optimality_cuts.append(optimality_cut)

            lower_bound = current_objective
        else:
            # Optimal solution found
            upper_bound = current_objective
            print("Optimal solution found (within tolerance).")
            break


    x_optimal = [x[i].x for i in range(n_x)]
    theta_optimal = theta.x if "theta" in locals() else 0
    return x_optimal, theta_optimal, lower_bound, upper_bound, iteration
```

**Explanation:**

1.  **Initialization:** Sets up the master problem with first-stage variables `x`, an auxiliary variable `theta` to represent the recourse function, and initial constraints.
2.  **Iteration:** The main loop iterates until a stopping criterion is met (maximum iterations or a sufficiently small gap between lower and upper bounds).
3.  **Master Problem Solution:** Solves the current master problem to obtain a candidate first-stage solution `x_val`.
4.  **Subproblem Solution:** For each scenario, a second-stage subproblem is solved with the first-stage variables fixed to the `x_val` obtained from the master problem.
5.  **Feasibility Cut Generation:** If a subproblem is infeasible, a feasibility cut is generated based on the dual variables of the infeasible subproblem's constraints. This cut is added to the master problem to exclude the current `x_val` from future consideration. *Important: The computation of the cut coefficients `gamma` and `delta` requires solving the dual problem of the infeasible subproblem.* The code checks the IIS of the infeasible subproblem.
6.  **Optimality Cut Generation:** If all subproblems are feasible, an optimality cut is generated. The cut approximates the recourse function `Q(x)` based on the dual variables of the subproblems' constraints. *Important: The computation of the cut coefficients `alpha` and `beta` requires solving the dual problem of the feasible subproblems.*
7.  **Stopping Criterion:**  The algorithm checks if the approximation of `Q(x)` is close enough. If so, it terminates.
8.  **Output:** Returns the optimal first-stage solution, the lower and upper bounds on the optimal objective value, and the number of iterations performed.

**Important Notes:**

*   **Dual Solution:** The generation of both optimality and feasibility cuts critically depends on solving the *dual* of the subproblems and extracting the dual variable values.  The provided code includes placeholders for these calculations.  You must implement the dualization and solution steps.
*   **Gurobi:** This implementation uses the Gurobi solver. You'll need a Gurobi license and the `gurobipy` package installed.
*   **Simplifications:** For clarity, the code omits certain common enhancements to the L-shaped method, such as:
    *   **Stabilization techniques:**  To improve convergence.
    *   **Pareto-optimality cuts:** To generate stronger cuts.
*   **Binary Variables**: The example assumes `x` are continuous for simplicity. If `x` are binary, `vtype=GRB.BINARY` should be used, and branch-and-cut is needed, making the algorithm more complex.
*   **Missing Data**: Replace the dummy data with your own stochastic linear program.

## 4) Follow-up question

How can the L-shaped method be extended to handle more than two stages of stochastic programming? What are the challenges and potential approaches?
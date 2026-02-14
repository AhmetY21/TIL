---
title: "Recourse Function Properties: convexity, piecewise linearity (linear recourse)"
date: "2026-02-14"
week: 7
lesson: 2
slug: "recourse-function-properties-convexity-piecewise-linearity-linear-recourse"
---

# Topic: Recourse Function Properties: convexity, piecewise linearity (linear recourse)

## 1) Formal definition (what is it, and how can we use it?)

In two-stage (or multi-stage) stochastic programming, the *recourse function* (also known as the second-stage value function) describes the optimal value of the second-stage problem given a particular realization of the uncertainty (i.e., a particular scenario) and a first-stage decision. Let's define it formally for a two-stage linear stochastic program:

Consider the following problem:

Minimize:  `c'x + E[Q(x, ω)]`

Subject to:  `Ax = b`
           `x >= 0`

Where:

*   `x` is the first-stage decision vector.
*   `ω` represents the random parameters (uncertainty).
*   `c` and `A` are cost and constraint matrices for the first stage.
*   `b` is the right-hand side vector for the first stage.
*   `E[Q(x, ω)]` is the expected recourse function, which represents the expected cost of taking corrective actions in the second stage after observing the realization of the uncertainty `ω`.

The recourse function, `Q(x, ω)`, is defined as:

`Q(x, ω) = min q(ω)'y`

Subject to:  `Wy = h(ω) - Tx`
           `y >= 0`

Where:

*   `y` is the second-stage decision vector (recourse action).
*   `q(ω)` is the cost vector for the second stage, dependent on the scenario `ω`.
*   `W` is the recourse matrix.
*   `T` is the technology matrix linking the first-stage and second-stage decisions.
*   `h(ω)` is the right-hand side vector for the second stage, dependent on the scenario `ω`.

**Convexity:** If the recourse problem is a linear program (as defined above, with linear objective and constraints), and the right-hand side, `h(ω) - Tx`, varies linearly with `x`, then the recourse function `Q(x, ω)` is a *convex* function of `x` for each fixed realization of `ω`.  This is a direct consequence of the optimality properties of linear programs. The minimum of a linear function subject to linear constraints is a convex function of parameters appearing in the constraints. Consequently, if `Q(x, ω)` is convex for each `ω`, then `E[Q(x, ω)]` is also convex (because the expectation operator preserves convexity). This is extremely useful, as it ensures that the overall stochastic programming problem maintains convexity if the first-stage problem is also convex.

**Piecewise Linearity (Linear Recourse):**  If `h(ω) - Tx` is a linear function of `x`, and `q(ω)`, `W` are independent of `x`, then `Q(x, ω)` is a *piecewise linear* function of `x`. This is because the feasible region of the second-stage problem can be partitioned into polyhedral regions, within each of which a particular set of basic variables remains optimal.  The optimal objective value `Q(x, ω)` is then a linear function of `x` within each of these polyhedral regions. This property is crucial for algorithms that exploit piecewise linearity, such as Benders decomposition and its variants. *Linear recourse* specifically refers to the situation where the second-stage problem is a linear program.

**How can we use it?**

*   **Algorithm Design:**  Convexity and piecewise linearity are essential properties for designing efficient solution algorithms for stochastic programs. Algorithms like Benders decomposition rely heavily on the piecewise linear nature of the recourse function.
*   **Approximation:**  If the recourse function is known to be convex and/or piecewise linear, we can use approximation techniques (e.g., cutting-plane methods) to efficiently approximate the expected recourse function `E[Q(x, ω)]`.
*   **Optimality Verification:** Knowing these properties can help in verifying the optimality of solutions obtained by different algorithms.
*   **Model Formulation:** Understanding these properties can guide the formulation of the stochastic programming model to ensure that desired properties like convexity are preserved, leading to more tractable problems.

## 2) Application scenario

Consider a supply chain management problem where a company needs to decide on the initial inventory level (`x`) of a product before knowing the exact demand (`ω`).  After observing the demand, the company can take recourse actions:

*   If demand exceeds inventory, they can order additional products at a higher cost (`q(ω)`) to satisfy the unmet demand.
*   If inventory exceeds demand, they can sell the excess inventory at a discounted price.

Here, the second-stage problem is to decide how much to order or sell based on the observed demand and the initial inventory level. The recourse function `Q(x, ω)` represents the cost associated with these recourse actions, which depends on the initial inventory level `x` and the realized demand `ω`.

Specifically:

*   `x`: Initial inventory level (first-stage decision).
*   `ω`: Random demand.
*   `y`: Second-stage decisions: `y1` = quantity ordered, `y2` = quantity sold.
*   `q(ω)`: Cost vector [cost of ordering, negative of salvage value].
*   The constraint is `x + y1 - y2 = ω`, which can be rewritten in the form `Wy = h(ω) - Tx` with appropriate definition of `W`, `h(ω)`, and `T`.

Because the second-stage decision is modeled as a linear program (minimizing cost of ordering/selling subject to demand constraints), the recourse function `Q(x, ω)` is convex and piecewise linear in `x` for each realization of `ω`.  This allows us to use efficient algorithms like Benders decomposition to find the optimal initial inventory level `x` that minimizes the total cost (initial inventory cost plus expected recourse cost).

## 3) Python method (if possible)

While there isn't a single, readily available function in a standard library to *directly* compute the recourse function `Q(x, ω)`, we can use optimization libraries like `cvxpy` or `gurobipy` to solve the second-stage problem for a given `x` and `ω`, thereby effectively evaluating the recourse function at that point. This can be incorporated into a broader stochastic programming solver.

Here's an example using `cvxpy` to evaluate the recourse function for a single scenario:

```python
import cvxpy as cp
import numpy as np

def evaluate_recourse_function(x_val, omega_val, q, W, T, h):
    """
    Evaluates the recourse function Q(x, omega) for a given x and omega.

    Args:
        x_val: Value of the first-stage decision variable x.
        omega_val: Realization of the random parameter omega.
        q: Cost vector for the second stage.
        W: Recourse matrix.
        T: Technology matrix.
        h: Right-hand side vector for the second stage (as a function of omega).

    Returns:
        The value of the recourse function Q(x, omega).
    """

    y = cp.Variable(W.shape[1], nonneg=True)  # Second-stage decision variable
    objective = q @ y
    constraints = [W @ y == h(omega_val) - T @ x_val]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL:
        return problem.value
    else:
        print("Second-stage problem is infeasible or unbounded.")
        return np.inf  # or some large value to indicate infeasibility


# Example usage:  Supply chain problem from the previous section.
# x: initial inventory level.  y = [y1, y2], where y1 = quantity ordered, y2 = quantity sold
# omega: demand

# Scenario parameters
omega_val = 10  # Demand

# First-stage decision (example)
x_val = np.array([7])

# Second-stage problem parameters
q = np.array([5, 0])  # Cost of ordering, salvage value is 0
W = np.array([[1, -1]])
T = np.array([1])  # Tx = x
h = lambda omega: np.array([omega])  # h(omega) = omega


# Evaluate the recourse function
Q_x_omega = evaluate_recourse_function(x_val, omega_val, q, W, T, h)
print(f"Recourse function value Q({x_val}, {omega_val}): {Q_x_omega}")

# To approximate E[Q(x, omega)], you would repeat this for multiple scenarios
# and average the results.
```

This code provides a way to numerically evaluate the recourse function `Q(x, ω)` for a given realization `ω` and first-stage decision `x`.  This allows for the calculation of the expected recourse cost, E[Q(x,ω)].

## 4) Follow-up question

How does the presence of integer variables in the second-stage (e.g., you can only order in whole units) impact the convexity and piecewise linearity properties of the recourse function? What algorithmic challenges arise from this, and what solution approaches can be used?
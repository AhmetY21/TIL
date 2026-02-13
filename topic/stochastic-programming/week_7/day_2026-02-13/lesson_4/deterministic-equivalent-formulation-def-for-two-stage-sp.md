---
title: "Deterministic Equivalent Formulation (DEF) for two-stage SP"
date: "2026-02-13"
week: 7
lesson: 4
slug: "deterministic-equivalent-formulation-def-for-two-stage-sp"
---

# Topic: Deterministic Equivalent Formulation (DEF) for two-stage SP

## 1) Formal definition (what is it, and how can we use it?)

The Deterministic Equivalent Formulation (DEF) of a two-stage stochastic program is a deterministic optimization problem that is equivalent to the original stochastic program.  By "equivalent," we mean that solving the DEF yields the optimal first-stage decisions, which, when implemented, lead to the same expected value of the objective function as solving the stochastic program directly (though solving the stochastic program "directly" is generally intractable unless the number of scenarios is very small).

Here's a breakdown:

*   **Two-Stage Stochastic Program:** These problems involve making decisions in two stages.
    *   **First Stage (Here-and-Now):**  Decisions made *before* the realization of uncertain parameters. Let's denote these variables as `x`.
    *   **Second Stage (Wait-and-See):** Decisions made *after* the realization of uncertain parameters. These decisions are recourse actions taken to mitigate the effect of the uncertainty.  Let's denote these variables as `y`. These second-stage variables depend on both the first-stage decisions `x` and the realization of the uncertain parameters, which we will denote with `ξ` (Greek letter xi), and thus are written as `y(x, ξ)`.
*   **Uncertainty:** The uncertain parameters (`ξ`) follow a known probability distribution. We often approximate this distribution by a discrete set of scenarios, denoted by `ω`. The probability of each scenario `ω` is denoted by `p_ω`.
*   **DEF Formulation:** The DEF explicitly incorporates all possible scenarios and their probabilities into a single, deterministic optimization problem.  Instead of having a second-stage decision function `y(x, ξ)`, we have specific second-stage decision variables, `y_ω`, for *each* scenario `ω`.  The objective function is the expected value of the total cost/profit across all scenarios.

**How can we use it?**

1.  **Solvability:** DEF transforms a difficult stochastic program into a (potentially very large) deterministic optimization problem that can be solved by standard optimization solvers (e.g., LP, MILP, QP, NLP solvers).
2.  **Tractability (Important caveat):** The size of the DEF grows linearly with the number of scenarios. If the uncertainty is represented by a large number of scenarios, the DEF can become computationally intractable due to its size. This is often referred to as the "curse of dimensionality."
3.  **Benchmarking:** If a solution to a smaller scenario set is found, it can be used as a benchmark or lower bound for more sophisticated solution approaches on the true (larger) scenario set.

**General Form of a Two-Stage SP and its DEF:**

*   **Two-Stage Stochastic Program:**

    ```
    min  c'x + E_ξ[Q(x, ξ)]
    s.t. Ax = b
         x >= 0
    ```

    where:
    *   `x` are the first-stage variables
    *   `c` is the cost vector for the first stage
    *   `Q(x, ξ) = min q(ξ)'y  s.t.  T(ξ)x + W(ξ)y = h(ξ), y >= 0` is the recourse function
    *   `ξ` represents the random parameters.
    *   `E_ξ[...]` denotes the expectation with respect to `ξ`.

*   **Deterministic Equivalent Formulation (using discrete scenarios):**

    ```
    min  c'x + Σ_ω p_ω * q_ω'y_ω
    s.t. Ax = b
         T_ω x + W_ω y_ω = h_ω,  ∀ ω ∈ Ω  (Ω is the set of scenarios)
         x >= 0
         y_ω >= 0, ∀ ω ∈ Ω
    ```

    where:
    *   `x` are the first-stage variables (same for all scenarios)
    *   `y_ω` are the second-stage variables for scenario `ω`
    *   `p_ω` is the probability of scenario `ω`
    *   `q_ω`, `T_ω`, `W_ω`, and `h_ω` are the values of `q(ξ)`, `T(ξ)`, `W(ξ)`, and `h(ξ)` under scenario `ω`

## 2) Application scenario

**Inventory Management Under Demand Uncertainty:**

A retailer needs to decide how much of a product to order from a supplier (first-stage decision: `x`: order quantity). The demand for the product is uncertain (random parameter: `ξ`). If the demand exceeds the order quantity, the retailer loses potential sales (lost sales cost). If the order quantity exceeds the demand, the retailer has to sell the remaining products at a discounted price (salvage value). The second-stage decision (`y_ω`) is how to handle the difference between the order quantity and the realized demand in each scenario `ω`.

*   **First-stage:** `x` = amount of product to order.
*   **Uncertainty:**  Demand for the product, `d_ω`, follows a discrete probability distribution (scenarios represent different possible demand levels).
*   **Second-stage:**
    *   `s_ω` = amount of product sold in scenario `ω`
    *   `ls_ω` = amount of lost sales in scenario `ω`
*   **Objective:** Minimize the sum of ordering costs and expected costs related to unsold inventory and lost sales.

**Mathematical formulation:**

Let `c` be the per-unit cost of ordering the product. Let `r` be the revenue per unit sold. Let `v` be the salvage value per unit of unsold inventory, and `pen` be the penalty cost per unit of lost sales.

The DEF would be:

```
min  c*x + Σ_ω p_ω * (r*s_ω - v*(x - s_ω) + pen*ls_ω)
s.t. s_ω + ls_ω = d_ω, ∀ ω
     s_ω <= x, ∀ ω
     x >= 0
     s_ω >= 0, ∀ ω
     ls_ω >= 0, ∀ ω
```

Here, we are minimizing the total cost. The first term is the ordering cost. The second term is the expected value (across all scenarios) of the revenue from sales, less salvage value of any remaining stock and cost of any lost sales.  The constraints in the `s.t.` (subject to) section ensure that the amount sold plus the amount of lost sales equals demand in each scenario, sales cannot exceed the order quantity, and all variables are non-negative.

## 3) Python method (if possible)

```python
import pyomo.environ as pyo
import numpy as np

def create_inventory_def(scenarios, probabilities, demands, c, r, v, pen):
    """
    Creates the deterministic equivalent formulation (DEF) for the inventory
    management problem.

    Args:
        scenarios (list): List of scenario names (e.g., ["Scenario1", "Scenario2"]).
        probabilities (list): List of probabilities for each scenario.
        demands (list): List of demand values for each scenario.
        c (float): Cost per unit of ordering the product.
        r (float): Revenue per unit sold.
        v (float): Salvage value per unit of unsold inventory.
        pen (float): Penalty cost per unit of lost sales.

    Returns:
        pyomo.environ.ConcreteModel: The Pyomo model representing the DEF.
    """

    model = pyo.ConcreteModel()

    # Sets
    model.SCENARIOS = pyo.Set(initialize=scenarios)

    # Parameters (can also be hardcoded within the constraints if desired)
    model.probabilities = pyo.Param(model.SCENARIOS, initialize=dict(zip(scenarios, probabilities)))
    model.demands = pyo.Param(model.SCENARIOS, initialize=dict(zip(scenarios, demands)))


    # Variables
    model.x = pyo.Var(domain=pyo.NonNegativeReals)  # Order quantity
    model.s = pyo.Var(model.SCENARIOS, domain=pyo.NonNegativeReals)  # Sales in each scenario
    model.ls = pyo.Var(model.SCENARIOS, domain=pyo.NonNegativeReals) # Lost sales in each scenario

    # Objective Function
    def objective_rule(model):
        return c * model.x + sum(model.probabilities[omega] * (r*model.s[omega] - v*(model.x - model.s[omega]) + pen*model.ls[omega]) for omega in model.SCENARIOS)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def demand_balance_rule(model, omega):
        return model.s[omega] + model.ls[omega] == model.demands[omega]
    model.demand_balance = pyo.Constraint(model.SCENARIOS, rule=demand_balance_rule)

    def sales_limit_rule(model, omega):
        return model.s[omega] <= model.x
    model.sales_limit = pyo.Constraint(model.SCENARIOS, rule=sales_limit_rule)

    return model

# Example Usage:
scenarios = ["LowDemand", "MediumDemand", "HighDemand"]
probabilities = [0.2, 0.5, 0.3]
demands = [50, 100, 150]
c = 10  # Ordering cost per unit
r = 20  # Revenue per unit sold
v = 5   # Salvage value per unit unsold
pen = 30 # Penalty per unit lost sale

model = create_inventory_def(scenarios, probabilities, demands, c, r, v, pen)

# Solve the model
solver = pyo.SolverFactory('glpk') # Or use another solver like 'gurobi', 'cplex'
solver.solve(model)

# Print the results
print(f"Optimal order quantity: {pyo.value(model.x)}")
for omega in scenarios:
    print(f"  Scenario {omega}: Sales = {pyo.value(model.s[omega])}, Lost Sales = {pyo.value(model.ls[omega])}")
print(f"Optimal objective value: {pyo.value(model.objective)}")
```

**Explanation:**

1.  **Pyomo Model:** We use Pyomo to define the optimization model.
2.  **Sets and Parameters:** We define sets (e.g., `SCENARIOS`) and parameters (e.g., probabilities, demands).
3.  **Variables:**  We define the decision variables: `x` (order quantity), `s_ω` (sales in each scenario), and `ls_ω` (lost sales in each scenario).
4.  **Objective Function:** The objective function minimizes the total expected cost.
5.  **Constraints:**  The constraints ensure that demand is met (either through sales or lost sales) and that sales do not exceed the order quantity.
6.  **Solving:** We use a solver (GLPK in this example, but you can use Gurobi or CPLEX if you have them installed) to solve the model.
7.  **Output:**  The code prints the optimal order quantity, sales, lost sales, and objective value.

## 4) Follow-up question

How does the Deterministic Equivalent Formulation relate to sample average approximation (SAA) in Stochastic Programming, and what are the pros and cons of using each method?
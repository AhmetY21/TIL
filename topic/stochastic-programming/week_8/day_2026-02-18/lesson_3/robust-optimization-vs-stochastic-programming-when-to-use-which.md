---
title: "Robust Optimization vs Stochastic Programming (when to use which)"
date: "2026-02-18"
week: 8
lesson: 3
slug: "robust-optimization-vs-stochastic-programming-when-to-use-which"
---

# Topic: Robust Optimization vs Stochastic Programming (when to use which)

## 1) Formal definition (what is it, and how can we use it?)

**Stochastic Programming (SP):**

Stochastic programming is a framework for modeling optimization problems where some of the problem parameters are uncertain and described by probability distributions.  Instead of solving a single deterministic problem, SP models consider a range of possible scenarios defined by these distributions. The goal is typically to find a solution that is optimal *in expectation* with respect to the uncertain parameters, or that minimizes a cost function that incorporates risk measures associated with the uncertainty.

Formally, a general two-stage stochastic programming problem can be written as:

`min_x {c'x + E_ξ [Q(x, ξ)]}`
`s.t. Ax = b`
`x ≥ 0`

Where:

*   `x` is the first-stage decision variable (decided before knowing the realization of the uncertainty).
*   `c` is the cost vector associated with the first-stage decision.
*   `ξ` is the random vector representing the uncertain parameters.
*   `E_ξ` denotes the expectation with respect to the probability distribution of `ξ`.
*   `Q(x, ξ)` is the *second-stage* (recourse) function representing the optimal value of a problem solved *after* observing the realization of `ξ`, given the first-stage decision `x`.  This second-stage problem corrects for any infeasibilities or inefficiencies introduced by the first-stage decision in light of the observed uncertainty.
*   `A` and `b` are constraint matrices and vectors, respectively.  These constraints apply to the first-stage decision `x`.

The key idea is to make decisions *now* (first-stage) knowing that we will have the opportunity to react *later* (second-stage) once the uncertainty is revealed.  We use the probability distribution of the uncertain parameters to guide our first-stage decision, aiming to minimize expected costs.  Multi-stage stochastic programming extends this concept to multiple time periods, where decisions are made sequentially as uncertainty unfolds.  Other variants can incorporate risk measures directly into the objective function (e.g., CVaR).

**Robust Optimization (RO):**

Robust optimization is another approach to dealing with optimization problems under uncertainty.  However, instead of relying on probability distributions, RO assumes that uncertain parameters belong to a *uncertainty set*.  The goal is to find a solution that is *feasible and optimal* for *all* possible realizations of the uncertain parameters within this set. This approach aims for *guaranteed feasibility and performance*, even in the worst-case scenario.

Formally, a generic robust optimization problem can be written as:

`min_x {max_{ξ ∈ U} f(x, ξ)}`
`s.t. g(x, ξ) ≤ 0, ∀ ξ ∈ U`

Where:

*   `x` is the decision variable.
*   `ξ` is the uncertain parameter.
*   `U` is the uncertainty set (defines the range of possible values for `ξ`).
*   `f(x, ξ)` is the objective function, which depends on both the decision `x` and the uncertain parameter `ξ`.
*   `g(x, ξ) ≤ 0` represents the constraints, which must hold for all possible realizations of `ξ` within the set `U`.

RO effectively optimizes for the worst-case scenario within the defined uncertainty set. This often leads to more conservative solutions compared to stochastic programming, but it guarantees feasibility and a certain level of performance regardless of the actual realization of the uncertain parameters. Different types of uncertainty sets (e.g., box, ellipsoidal, polyhedral) can be used to model different assumptions about the uncertainty.

**When to Use Which:**

*   **Stochastic Programming:** Use when you have a good understanding of the probability distribution of the uncertain parameters. SP allows you to balance performance and robustness by optimizing for expected outcomes and incorporating risk measures.  It is suitable when accepting a small probability of infeasibility or sub-optimality is acceptable in exchange for better average performance. Best used when a reasonable probabilistic model of the uncertainty exists.

*   **Robust Optimization:** Use when you lack reliable probabilistic information about the uncertain parameters, or when guaranteed feasibility and performance are critical, even under worst-case scenarios.  RO provides a hedge against extreme events and is suitable when you absolutely must avoid infeasibility or significant performance degradation, regardless of how unlikely a scenario might be. Appropriate when a precise probabilistic model is not available, or one is not trusted, and a certain level of conservatism is acceptable to ensure constraint satisfaction.  RO is also useful when the problem requires decisions that are valid under all circumstances, even extreme ones.

In summary:

| Feature         | Stochastic Programming          | Robust Optimization                 |
|-----------------|---------------------------------|--------------------------------------|
| Uncertainty Model | Probability distributions       | Uncertainty sets                     |
| Optimality       | Optimal in expectation           | Optimal in the worst-case scenario   |
| Feasibility     | May be infeasible in some scenarios | Always feasible (within the uncertainty set) |
| Conservatism      | Less conservative               | More conservative                    |
| Data Requirements | Requires probability distributions | Requires defining uncertainty sets |
| Computational Complexity | Can be high, especially for multi-stage problems | Can be high, depends on the uncertainty set structure |
| Use Cases       | Inventory management, finance, energy markets, where historical data is available | Supply chain design, engineering design, safety-critical systems, where robustness is paramount |

## 2) Application scenario

**Stochastic Programming Application:**

Consider an inventory management problem where a company needs to decide how much of a product to order each month.  The demand for the product is uncertain, but the company has historical data that allows it to estimate the probability distribution of demand for each month.  Using stochastic programming, the company can formulate a multi-stage stochastic program that considers different demand scenarios. The first-stage decision is the order quantity.  The second-stage decision is how to handle excess or shortage of inventory based on the realized demand (e.g., backorders, holding costs). The objective is to minimize the expected total cost, including ordering costs, holding costs, and backorder costs, across all scenarios.

**Robust Optimization Application:**

Now consider a power grid operator who needs to decide on the dispatch schedule for power plants.  The renewable energy generation from wind and solar is uncertain, but the operator does not have reliable probability distributions for these sources.  Instead, the operator defines an uncertainty set that represents the possible range of wind and solar generation. Using robust optimization, the operator formulates a problem that ensures the power grid can meet demand under all possible realizations of wind and solar generation within the defined uncertainty set. The objective is to minimize the operating cost of the power plants, subject to constraints on power balance, generator capacity, and transmission line limits, which must hold for all possible scenarios.

## 3) Python method (if possible)

Here's a simple example using the `pyomo` library for solving a deterministic linear program and then illustrating how you might *conceptually* implement aspects of both Stochastic Programming and Robust Optimization. This isn't a full implementation due to the complexity of these topics, but it demonstrates the basic structure.

```python
import pyomo.environ as pyo
import numpy as np

# Deterministic Linear Program
def solve_deterministic_lp(c, A, b):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(A.shape[1]), within=pyo.NonNegativeReals)

    model.objective = pyo.Objective(expr=sum(c[i] * model.x[i] for i in range(A.shape[1])), sense=pyo.minimize)

    model.constraints = pyo.ConstraintList()
    for i in range(A.shape[0]):
        model.constraints.add(sum(A[i, j] * model.x[j] for j in range(A.shape[1])) <= b[i])

    solver = pyo.SolverFactory('glpk')  # You can use other solvers like 'cplex' or 'gurobi' if installed
    solver.solve(model)

    return [pyo.value(model.x[i]) for i in range(A.shape[1])]


# Conceptual Stochastic Programming (Scenario-based)
def conceptual_stochastic_programming(c, A, b, scenario_probabilities, scenario_parameters): #parameters can be c, A, or b
    #This is a highly simplified illustration.  True SP requires solving multiple linked problems.
    #We'll just simulate evaluating expected cost.
    expected_cost = 0
    for i, prob in enumerate(scenario_probabilities):
        #Simulate solving the problem for each scenario
        scenario_c = scenario_parameters[i]  #For simplicity, assume uncertain cost only
        x = solve_deterministic_lp(scenario_c, A, b)
        scenario_cost = np.dot(scenario_c, x)
        expected_cost += prob * scenario_cost

    return expected_cost

# Conceptual Robust Optimization (Interval Uncertainty)
def conceptual_robust_optimization(c_nominal, A, b, c_uncertainty_range):
    #Worst-case objective value, we maximize objective value with biggest cost vector
    #Find the most unfavorable objective for a fixed x.  This is not a complete RO solver.

    c_worst = c_nominal + c_uncertainty_range  # Assuming interval uncertainty of [0, c_uncertainty_range]
    x = solve_deterministic_lp(c_worst, A, b)
    worst_case_cost = np.dot(c_worst, x)
    return worst_case_cost

# Example Usage
if __name__ == '__main__':
    # Deterministic Problem Data
    c = np.array([1, 2])
    A = np.array([[1, 1], [2, 1]])
    b = np.array([3, 4])

    # Solve Deterministic Problem
    x_deterministic = solve_deterministic_lp(c, A, b)
    print("Deterministic Solution:", x_deterministic)

    # Stochastic Programming Data (Simplified)
    scenario_probabilities = [0.6, 0.4]
    scenario_parameters = [np.array([1.2, 2.2]), np.array([0.8, 1.8])] #Scenario c values
    expected_cost = conceptual_stochastic_programming(c, A, b, scenario_probabilities, scenario_parameters)
    print("Conceptual Stochastic Programming Expected Cost:", expected_cost)


    # Robust Optimization Data (Simplified)
    c_uncertainty_range = np.array([0.2, 0.3]) #Uncertainty is +0.2, +0.3 to nominal c
    worst_case_cost = conceptual_robust_optimization(c, A, b, c_uncertainty_range)
    print("Conceptual Robust Optimization Worst-Case Cost:", worst_case_cost)
```

Key points:

*   **Pyomo:**  Pyomo is a powerful Python library for modeling and solving optimization problems.  It provides a high-level interface for defining optimization models and interfacing with various solvers.
*   **Deterministic LP:**  The `solve_deterministic_lp` function solves a standard linear program using Pyomo and the GLPK solver. You would replace 'glpk' with 'cplex' or 'gurobi' if you have those solvers installed.
*   **Stochastic Programming (Conceptual):** The `conceptual_stochastic_programming` function is *highly simplified*.  A true stochastic program requires solving a different optimization problem for each scenario and linking them together through first-stage variables. This code just demonstrates how you would calculate an *expected cost* if you *already had* the solution for each scenario. Pyomo does have extensions for multistage stochastic programming, but implementation is involved.  Real stochastic programming problems involve much more complex formulations than simply solving a deterministic LP for each scenario and averaging.
*   **Robust Optimization (Conceptual):** The `conceptual_robust_optimization` function also provides a simplified illustration. The worst-case scenario is found and the resulting cost is calculated. Again, this is not a full RO solver, which would require solving a more complex problem that optimizes the decision variable `x` while simultaneously considering the worst-case realization of the uncertain parameters. Implementating true RO often involves reformulation techniques to deal with the min-max structure of the problem and the specific properties of the uncertainty set.

Important notes:

*   **Solvers:** To run the code, you'll need to have a solver like GLPK, CPLEX, or Gurobi installed and configured with Pyomo.
*   **Simplifications:** The stochastic and robust optimization examples are significantly simplified.  A full implementation would involve much more complex formulations and require specialized solution techniques.
*   **Real-world problems:** Real-world stochastic and robust optimization problems can be very challenging to solve, especially for large-scale instances.

## 4) Follow-up question

Given a problem where the uncertain parameter is the demand for a product and historical data suggests the demand follows a normal distribution but with potential outliers due to unexpected market events, how would you choose between stochastic programming and robust optimization, and what specific modeling choices would you make within each framework to address this situation? For example, what risk measure might be relevant in the stochastic programming case? And what type of uncertainty set could be appropriate for robust optimization?
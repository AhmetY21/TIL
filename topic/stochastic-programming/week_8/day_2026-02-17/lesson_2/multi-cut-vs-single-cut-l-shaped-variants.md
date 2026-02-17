---
title: "Multi-Cut vs Single-Cut L-shaped Variants"
date: "2026-02-17"
week: 8
lesson: 2
slug: "multi-cut-vs-single-cut-l-shaped-variants"
---

# Topic: Multi-Cut vs Single-Cut L-shaped Variants

## 1) Formal definition (what is it, and how can we use it?)

The L-shaped method is a decomposition technique used to solve two-stage stochastic programming problems. These problems involve making decisions now (first-stage decisions) before knowing the realization of uncertain parameters. After observing the uncertainty (e.g., demand, weather), we make adjustments (second-stage decisions) to mitigate the impact of the first-stage decisions and the realized uncertainty. The L-shaped method reformulates the problem by iteratively building outer approximations (cuts) of the second-stage value function.

**Single-Cut L-shaped method:** In the single-cut variant, each iteration generates only *one* cut that approximates the second-stage recourse function. This cut represents a linear lower bound on the expected value of the second-stage problem, and it is valid for all possible realizations of the uncertain parameters (i.e., the scenarios). The cut typically looks like:

```
theta >= alpha + beta' * x
```

where:

*   `theta` is an auxiliary variable representing the optimal value of the second-stage problem.
*   `x` is the vector of first-stage decision variables.
*   `alpha` is a constant representing the intercept of the cut.
*   `beta` is a vector representing the slope of the cut with respect to `x`.
*   `'` denotes the transpose.

This single cut is added to the master problem in each iteration. While conceptually simpler, the single-cut method can converge slowly, especially when the second-stage problem is non-convex or has a large feasible region for the first-stage variables.

**Multi-Cut L-shaped method:** In the multi-cut variant, *each scenario* (realization of the uncertain parameters) generates its own cut. These cuts are scenario-specific lower bounds on the second-stage problem's optimal value. Instead of a single auxiliary variable `theta`, each scenario `omega` has its own auxiliary variable `theta_omega`. The cuts look like:

```
theta_omega >= alpha_omega + beta_omega' * x   for each scenario omega
```

The master problem then considers the expected value of these scenario-specific variables `theta_omega`.  The master problem now contains scenario-specific cuts. This approach generally leads to a tighter outer approximation of the value function compared to the single-cut version and, consequently, faster convergence.

The main difference is the *granularity* of the approximation. Single-cut provides a single lower bound on the *expected* recourse function, while multi-cut provides separate lower bounds for the recourse function under *each* scenario.  This finer representation leads to potentially better solutions and faster convergence, but at the cost of more variables and constraints in the master problem.

**How to use it:** Both variants are used to decompose the original two-stage stochastic programming problem into a master problem (first-stage decision) and a subproblem (second-stage recourse for each scenario). They are used iteratively.
1. Solve master problem: Obtain a first-stage solution `x`.
2. Solve subproblems: For each scenario, solve the second-stage problem using the `x` obtained from the master problem.
3. Generate cuts:  Based on the solution of subproblems, generate either a single cut or multiple cuts depending on which variant of the algorithm is being used.
4. Add cuts to the master problem and repeat from step 1 until convergence.

## 2) Application scenario

Consider a power generation planning problem. A power company needs to decide how much capacity to invest in various types of power plants (e.g., coal, gas, solar) *before* knowing the exact future electricity demand. The future demand is uncertain, and there are different demand scenarios (e.g., high, medium, low).

*   **First-stage decision (x):** Capacity investment in each type of power plant.
*   **Uncertain parameter (omega):** Electricity demand in different regions.
*   **Second-stage decision (y):** Power generation levels from each plant to meet demand and minimize operational costs, given the capacity investment and the realized demand.

In this scenario:

*   **Single-cut:** Would generate a single cut approximating the *expected* cost of meeting demand across all scenarios, given the capacity investment.
*   **Multi-cut:** Would generate a separate cut for each demand scenario. For example, a high demand scenario might generate a cut indicating that the current investment is insufficient to meet demand efficiently, thus requiring more expensive generation.  The low demand scenario might generate a cut indicating that too much capacity has been built, increasing fixed costs.

The multi-cut approach will likely provide a more accurate representation of the cost of meeting demand under different scenarios, leading to a better investment decision.  It allows the model to recognize and penalize the specific consequences of undershooting or overshooting capacity in each scenario. A single cut might average out these effects and lead to a suboptimal first-stage decision.

## 3) Python method (if possible)

While a complete implementation of L-shaped method is quite lengthy, here's a simplified example showcasing how to generate *cuts* using the `Pyomo` library, assuming you already have the solutions from the second-stage problem:

```python
import pyomo.environ as pyo
import numpy as np

def generate_single_cut(x_val, Q_omega_vals, scenario_probabilities):
  """Generates a single-cut based on solutions from all scenarios.
     x_val: numpy array of first-stage decision variable values.
     Q_omega_vals: Dictionary, where keys are scenario names and values are the optimal second-stage objective values for that scenario.
     scenario_probabilities: Dictionary, scenario probabilities
  """

  # Calculate the expected second-stage cost
  expected_cost = sum(scenario_probabilities[omega] * Q_omega_vals[omega] for omega in Q_omega_vals)

  # Calculate the beta vector (derivative of the second-stage cost with respect to x)
  # This is a simplification! In practice, calculating beta involves dual variables
  # from the second-stage problem.  We're assuming here it is pre-calculated.

  # For demonstration purposes, let's assume beta is a constant vector
  beta = np.array([0.1, 0.2])  # Example slope values

  # Calculate alpha (intercept)
  alpha = expected_cost - np.dot(beta, x_val)

  return alpha, beta


def generate_multi_cut(x_val, Q_omega_vals, scenario_probabilities):
    """Generates multiple cuts, one for each scenario.
       x_val: numpy array of first-stage decision variable values.
       Q_omega_vals: Dictionary, where keys are scenario names and values are the optimal second-stage objective values for that scenario.
       scenario_probabilities: Dictionary, scenario probabilities
    """

    cuts = {}
    for omega in Q_omega_vals:
        # Calculate the beta vector (derivative of the second-stage cost with respect to x)
        # This is a simplification! In practice, calculating beta involves dual variables
        # from the second-stage problem.  We're assuming here it is pre-calculated.

        # For demonstration purposes, let's assume beta is a constant vector
        beta_omega = np.array([0.1, 0.2])  # Example slope values

        # Calculate alpha (intercept)
        alpha_omega = Q_omega_vals[omega] - np.dot(beta_omega, x_val)

        cuts[omega] = (alpha_omega, beta_omega)

    return cuts


# Example usage
x_val = np.array([5, 10]) # Example first-stage solution
Q_omega_vals = {"scenario_1": 15, "scenario_2": 20, "scenario_3": 18} # Example second-stage costs
scenario_probabilities = {"scenario_1": 0.3, "scenario_2": 0.4, "scenario_3": 0.3} # Example probabilities

# Generate single cut
alpha_single, beta_single = generate_single_cut(x_val, Q_omega_vals, scenario_probabilities)
print(f"Single Cut: theta >= {alpha_single} + {beta_single}' * x")

# Generate multi cuts
cuts_multi = generate_multi_cut(x_val, Q_omega_vals, scenario_probabilities)
for omega, (alpha_omega, beta_omega) in cuts_multi.items():
    print(f"Multi Cut ({omega}): theta_{omega} >= {alpha_omega} + {beta_omega}' * x")
```

**Important Notes:**

*   **Dual Variables:** The code above significantly *simplifies* the cut generation process. In a real L-shaped implementation, `beta` is typically derived from the dual variables of the second-stage problem at optimality. The dual variables provide sensitivity information, indicating how much the second-stage objective function changes with respect to changes in the first-stage variables. Calculating these duals is a key part of implementing the L-shaped method.
*   **Master Problem:** The generated cuts (either single or multiple) would then be *added* as constraints to the master problem. This involves creating Pyomo model components (variables like `theta` or `theta_omega` and constraints based on the `alpha` and `beta` values).
*   **Iteration:** This process of solving the master problem, solving the second-stage problems for each scenario, generating cuts, and adding them back to the master problem is repeated iteratively until convergence.
*   **Implementation Complexity:** Implementing the full L-shaped method requires careful management of the master problem and subproblems, including updating the model, solving it, and extracting the necessary information. The choice between single-cut and multi-cut will depend on the specific problem structure and the computational cost of solving the master problem with many cuts versus the potential for faster convergence.

## 4) Follow-up question

Under what circumstances would using a single-cut L-shaped method be preferable to a multi-cut L-shaped method, given that multi-cut generally provides a tighter relaxation? Consider computational complexity and problem structure.
---
title: "Sample Average Approximation (SAA): replacing expectation with sample mean"
date: "2026-02-14"
week: 7
lesson: 6
slug: "sample-average-approximation-saa-replacing-expectation-with-sample-mean"
---

# Topic: Sample Average Approximation (SAA): replacing expectation with sample mean

## 1) Formal definition (what is it, and how can we use it?)

Sample Average Approximation (SAA) is a technique used to approximate stochastic programming problems where the objective function or constraints involve expectations. The core idea is to replace the intractable expectation operator with a sample average calculated from a finite number of independent and identically distributed (i.i.d.) samples of the random variables.

Formally, consider a stochastic program of the form:

```
min  f(x) = E[F(x, ξ)]
s.t.  x ∈ X
       G(x, ξ) ≤ 0  (or potentially other constraint forms),
```

where:

*   `x` is the decision variable.
*   `ξ` is a random variable (or a vector of random variables) with a known probability distribution.
*   `E[F(x, ξ)]` represents the expected value of the function `F(x, ξ)` with respect to the random variable `ξ`. This is often the "true" objective function we want to minimize.
*   `X` is a feasible set (e.g., box constraints, linear constraints).
*   `G(x, ξ) ≤ 0` represents stochastic constraints, which must be satisfied for all (or most) realizations of `ξ`.

The expectation `E[F(x, ξ)]` is typically difficult or impossible to compute analytically. SAA approximates this expectation by generating a sample of `N` independent realizations of the random variable `ξ`, denoted by `ξ₁, ξ₂, ..., ξN`. We then replace the expectation with the sample average:

```
E[F(x, ξ)] ≈ (1/N) * Σᵢ F(x, ξᵢ)  (for i = 1 to N)
```

The SAA problem becomes:

```
min  f_N(x) = (1/N) * Σᵢ F(x, ξᵢ)
s.t.  x ∈ X
       G(x, ξᵢ) ≤ 0  for all i = 1 to N
```

where `f_N(x)` is the sample average objective function.

**How to use it:**

1.  **Identify the stochastic program:** Define the objective function, constraints, decision variables, and the random variable(s) with their distributions.
2.  **Generate a sample:** Draw `N` i.i.d. samples `ξ₁, ξ₂, ..., ξN` from the distribution of `ξ`.  The choice of `N` is critical and depends on the problem's characteristics and desired accuracy. Larger `N` generally leads to better approximations but higher computational cost.
3.  **Formulate the SAA problem:** Replace the expectation in the original stochastic program with the sample average, creating a deterministic optimization problem.
4.  **Solve the SAA problem:** Use standard optimization algorithms (e.g., linear programming, mixed-integer programming, nonlinear programming solvers) to find the optimal solution `x_N` of the SAA problem.
5.  **Assess the solution:** Validate the solution `x_N` by, for example, evaluating its performance with another independent sample or using statistical tests to estimate the optimality gap. Increasing the sample size N and re-solving can often improve solution quality.

SAA transforms a computationally challenging stochastic problem into a solvable deterministic one, albeit an approximation.  The quality of the approximation depends on the sample size `N` and the properties of the function `F(x, ξ)` and `G(x, ξ)`.

## 2) Application scenario

Consider a portfolio optimization problem where an investor wants to minimize the risk (e.g., variance or conditional value-at-risk) of their portfolio subject to a target expected return.  The returns of the assets are uncertain and are modeled as random variables.

**Original stochastic program:**

```
min  CVaR_α(x, ξ)  (Conditional Value-at-Risk at level α)
s.t.  E[r'x] >= R_target (Expected return constraint)
       Σᵢ xᵢ = 1        (Budget constraint: sum of weights equals 1)
       xᵢ >= 0          (Non-negativity: no short selling)
```

where:

*   `x` is the vector of portfolio weights (decision variables).
*   `ξ` represents the random asset returns (a vector of random variables, `r`).
*   `CVaR_α` is the Conditional Value-at-Risk at level `α`, measuring the expected loss exceeding the Value-at-Risk. The calculation involves an expectation over scenarios.  `r'x` is the portfolio return.
*   `E[r'x]` is the expected return of the portfolio.
*   `R_target` is the target expected return.

**SAA application:**

1.  We generate `N` scenarios of asset returns `r₁, r₂, ..., rN` from a suitable distribution (e.g., multivariate normal distribution fit to historical data, or simulated using Monte Carlo methods).
2.  We replace the expectation in the expected return constraint and the CVaR calculation with sample averages:

```
min  CVaR_α_N(x) ≈ (1/N) * Σᵢ indicator_function(rᵢ'x <= VaR_α) * (VaR_α - rᵢ'x) (SAA approximation of CVaR)
s.t.  (1/N) * Σᵢ rᵢ'x >= R_target (SAA approximation of expected return constraint)
       Σᵢ xᵢ = 1
       xᵢ >= 0
```

Where `CVaR_α_N` is the sample average approximation of the CVaR.  The indicator function in the CVaR approximation can be handled with auxiliary variables, and the resulting problem is often a linear program or a mixed-integer program depending on the chosen CVaR implementation.

By solving this SAA problem, we obtain an approximate optimal portfolio `x_N`. This approach allows us to handle the uncertainty in asset returns by using a set of plausible scenarios to drive the optimization process. The quality of the portfolio depends on the chosen sample size N.

## 3) Python method (if possible)

```python
import numpy as np
import cvxpy as cp

def saa_portfolio_optimization(returns, R_target, alpha=0.05):
  """
  Performs portfolio optimization using Sample Average Approximation (SAA).

  Args:
    returns: A numpy array of shape (N, M) representing asset returns, where N is the number of scenarios and M is the number of assets.
    R_target: The target expected return.
    alpha: The confidence level for CVaR (e.g., 0.05 for 95% confidence).

  Returns:
    A numpy array of portfolio weights.
  """

  N, M = returns.shape

  # Define decision variables
  x = cp.Variable(M)  # Portfolio weights
  z = cp.Variable()  # Value-at-Risk (VaR)

  # Define auxiliary variables for CVaR approximation
  y = cp.Variable(N, nonneg=True)

  # Define the objective function (SAA of CVaR)
  cvar = z + (1 / (alpha * N)) * cp.sum(y)

  # Define constraints
  constraints = [
      cp.sum(returns @ x) / N >= R_target,  # SAA of expected return constraint
      cp.sum(x) == 1,                       # Budget constraint
      x >= 0,                                 # Non-negativity constraint
      returns @ x <= z + y,                  # CVaR auxiliary constraints
      y >= 0
  ]

  # Define the problem
  problem = cp.Problem(cp.Minimize(cvar), constraints)

  # Solve the problem
  problem.solve()

  # Return the optimal portfolio weights
  if problem.status == cp.OPTIMAL:
    return x.value
  else:
    print("Optimization failed:", problem.status)
    return None


# Example usage
if __name__ == '__main__':
  # Generate some random return data (replace with your actual data)
  np.random.seed(42)
  N = 100  # Number of scenarios
  M = 5    # Number of assets
  returns = np.random.normal(0.1, 0.2, size=(N, M))  # Mean 0.1, std dev 0.2

  R_target = 0.15  # Target expected return

  # Perform portfolio optimization using SAA
  optimal_weights = saa_portfolio_optimization(returns, R_target)

  if optimal_weights is not None:
    print("Optimal portfolio weights:", optimal_weights)
    print("Sum of weights:", np.sum(optimal_weights)) # should be close to 1
    portfolio_return = np.mean(returns @ optimal_weights)
    print(f"Estimated portfolio return using SAA: {portfolio_return}")
```

**Explanation:**

1.  **`saa_portfolio_optimization(returns, R_target, alpha)`:**  This function takes asset returns (a NumPy array), the target return, and the CVaR confidence level as input.
2.  **Decision Variables:** `x` represents the portfolio weights, `z` represents VaR and `y` contains auxiliary variables for CVaR calculation.
3.  **Objective Function:** The objective is to minimize CVaR (approximated using SAA).
4.  **Constraints:**
    *   `cp.sum(returns @ x) / N >= R_target`: The sample average of the portfolio return must be greater than or equal to the target return.
    *   `cp.sum(x) == 1`: The portfolio weights must sum to 1 (budget constraint).
    *   `x >= 0`: No short-selling allowed (non-negativity constraint).
    *   `returns @ x <= z + y`: This is one standard way to express the CVaR constraints, which along with `y >= 0` ensures we penalize values exceeding VaR.
5.  **CVXPY:**  This example uses the `cvxpy` library for convex optimization in Python. It's a user-friendly library for formulating and solving optimization problems.
6.  **Example Usage:** The `if __name__ == '__main__':` block provides a basic example of how to use the function.  It generates random return data, sets a target return, and calls `saa_portfolio_optimization`. It then prints the optimal portfolio weights and verifies that they sum to 1.

**Important notes:**

*   This is a simplified example. In a real-world application, you would use actual historical return data or more sophisticated models for generating scenarios.
*   The quality of the SAA solution depends on the sample size (`N`). You should experiment with different values of `N` to find a balance between accuracy and computational cost.
*   The `cvxpy` library needs to be installed (`pip install cvxpy`).
*   This example uses the CVaR approximation.  You could also use other risk measures or different formulations.
* The example uses a mixed integer approach to address the CVaR, but alternative linear programming approaches also exist

## 4) Follow-up question

How does the choice of sample size *N* in SAA impact the solution's quality and the computational cost, and what strategies can be used to determine a suitable value for *N* for a given problem?
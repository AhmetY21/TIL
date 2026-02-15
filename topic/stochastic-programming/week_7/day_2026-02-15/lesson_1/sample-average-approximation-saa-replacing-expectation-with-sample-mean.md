---
title: "Sample Average Approximation (SAA): replacing expectation with sample mean"
date: "2026-02-15"
week: 7
lesson: 1
slug: "sample-average-approximation-saa-replacing-expectation-with-sample-mean"
---

# Topic: Sample Average Approximation (SAA): replacing expectation with sample mean

## 1) Formal definition (what is it, and how can we use it?)

Sample Average Approximation (SAA) is a method used to approximate stochastic optimization problems, which are optimization problems that involve uncertainty in their objective function or constraints. The core idea of SAA is to replace the expectation in the stochastic program with a sample average calculated from a finite set of random samples.

More formally, consider the stochastic optimization problem:

```
minimize  F(x) = E[f(x, ξ)]
subject to x ∈ X
```

where:
*   `x` is the decision variable
*   `ξ` is a random variable following a probability distribution P
*   `f(x, ξ)` is the stochastic objective function
*   `E[f(x, ξ)]` is the expected value of the objective function with respect to the random variable `ξ`
*   `X` is the feasible region of the decision variable `x`

Directly solving this problem is often computationally intractable because evaluating the expectation `E[f(x, ξ)]` can be difficult or impossible. SAA addresses this by approximating the expectation using a Monte Carlo sample. We generate `N` independent and identically distributed (i.i.d.) samples `ξ₁, ξ₂, ..., ξₙ` from the distribution P. Then, the SAA approximation of the objective function is:

```
Fₙ(x) = (1/N) * Σᵢ₌₁ⁿ f(x, ξᵢ)
```

The SAA problem becomes:

```
minimize  Fₙ(x) = (1/N) * Σᵢ₌₁ⁿ f(x, ξᵢ)
subject to x ∈ X
```

We then solve this deterministic approximation. The optimal solution `xₙ*` of the SAA problem is an approximation of the optimal solution of the original stochastic problem.

**How we can use it:**

1.  **Approximate hard problems:** SAA allows us to solve stochastic optimization problems that are intractable due to the presence of expectations.
2.  **Estimate solution quality:** We can use statistical properties of the SAA solution, such as confidence intervals, to assess the quality of the approximate solution.  As the sample size `N` increases, the SAA problem converges (under certain conditions) to the true stochastic problem.
3.  **Scenario reduction:** SAA can be seen as a scenario reduction technique where an infinite or extremely large number of potential scenarios are reduced to a finite number based on the sampled observations.
4. **Simulation and Optimization:** Allows combining simulation to generate the samples with optimization techniques to find good solutions under uncertainty.

## 2) Application scenario

A classic application scenario is **portfolio optimization under uncertainty**. Consider an investor wanting to allocate capital across different assets.  The future returns of these assets are uncertain.  The investor wants to minimize risk (e.g., variance) while achieving a target return.

The problem can be formulated as:

```
minimize  E[(rᵀx - μ)²]  (Minimize the variance of portfolio return)
subject to:
E[rᵀx] >= R  (Achieve a target expected return R)
Σᵢ xᵢ = 1   (Budget constraint: all capital must be invested)
xᵢ >= 0   (No short selling)
```

where:
*   `x` is a vector of investment proportions for each asset
*   `r` is a vector of random asset returns
*   `μ` is the mean vector of random asset returns
* `R` is the targeted return

Here, the expectation is with respect to the uncertain asset returns `r`. We don't know the true distribution of `r`. Using SAA, we can:

1.  Generate `N` samples of possible asset return scenarios (e.g., using historical data or simulation models).  This gives us `r₁, r₂, ..., rₙ`.
2.  Replace the expectations with sample averages. For example:
    *   `E[(rᵀx - μ)²]` is approximated by `(1/N) * Σᵢ₌₁ⁿ (rᵢᵀx - (1/N)*Σⱼ₌₁ⁿ rⱼᵀx)²`
    *   `E[rᵀx]` is approximated by `(1/N) * Σᵢ₌₁ⁿ rᵢᵀx`

The resulting SAA problem is a deterministic quadratic program, which can be solved using standard optimization solvers. The optimal investment proportions `x*` obtained from the SAA problem provide an approximate solution to the original stochastic portfolio optimization problem.

## 3) Python method (if possible)

Here's a Python example using `cvxpy` to implement SAA for a simplified portfolio optimization problem:

```python
import numpy as np
import cvxpy as cp

def solve_portfolio_saa(returns, target_return):
    """
    Solves a portfolio optimization problem using Sample Average Approximation (SAA).

    Args:
        returns (np.ndarray): A matrix where each row represents a sample of asset returns.
                             Shape: (N, num_assets), where N is the number of samples.
        target_return (float): The desired target return.

    Returns:
        np.ndarray: The optimal portfolio allocation (vector of investment proportions).
    """

    num_assets = returns.shape[1]
    x = cp.Variable(num_assets)  # Investment proportions

    # Objective: Minimize portfolio variance (using the sample average approximation)
    portfolio_return = returns @ x
    variance = cp.sum_squares(portfolio_return - cp.sum(portfolio_return) / returns.shape[0]) / returns.shape[0]
    objective = cp.Minimize(variance)  # Minimize variance

    # Constraints:
    constraints = [
        cp.sum(x) == 1,  # Budget constraint
        x >= 0,          # No short selling
        cp.sum(portfolio_return) / returns.shape[0] >= target_return  # Target return
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL:
        return x.value
    else:
        print("Problem is infeasible or unbounded")
        return None


if __name__ == '__main__':
    # Example usage:
    np.random.seed(42)  # for reproducibility
    num_samples = 100
    num_assets = 5
    returns = np.random.normal(0.1, 0.05, size=(num_samples, num_assets))  # Simulate asset returns

    target_return = 0.12

    optimal_portfolio = solve_portfolio_saa(returns, target_return)

    if optimal_portfolio is not None:
        print("Optimal Portfolio Allocation:", optimal_portfolio)
        print("Expected Return (SAA):", np.mean(returns @ optimal_portfolio))
```

**Explanation:**

1.  **`solve_portfolio_saa(returns, target_return)` function:**
    *   Takes asset return samples (`returns`) and a target return (`target_return`) as input.
    *   Defines `x` as a `cvxpy` variable representing the investment proportions.
    *   Constructs the objective function: Minimize portfolio variance, approximated using the sample average.
    *   Defines the constraints: budget constraint (sum of `x` equals 1), no short selling (`x >= 0`), and a constraint ensuring the portfolio's expected return (approximated via sample average) meets the target.
    *   Solves the `cvxpy` optimization problem.
    *   Returns the optimal portfolio allocation if a solution is found.

2.  **`if __name__ == '__main__':` block:**
    *   Sets up an example by generating random asset return samples.
    *   Calls `solve_portfolio_saa` to find the optimal portfolio.
    *   Prints the results.  It also calculates and prints the expected return of the resulting portfolio based on the SAA samples to assess how well it achieved the desired target.

**Important considerations:**

*   **Sample Size (N):** The accuracy of the SAA solution depends heavily on the sample size `N`.  A larger `N` generally leads to a better approximation, but also increases the computational cost. Determining the appropriate `N` often involves statistical analysis to balance accuracy and computational effort.
*   **Solver Choice:**  The choice of optimization solver (e.g., the solver used by `cvxpy`) can also affect the speed and accuracy of the solution.
*   **Distribution of Random Variables:**  The convergence properties of SAA depend on the properties of the distribution of the random variables.  For example, strong convexity and Lipschitz continuity of the objective function can guarantee convergence.
* **Statistical Guarantees:**  SAA provides statistical guarantees on the quality of the solution.  Techniques like sample path analysis and confidence interval estimation can be used to assess the solution's optimality gap (the difference between the SAA solution and the true optimal solution).

## 4) Follow-up question

How can we improve the efficiency of SAA, especially when dealing with computationally expensive objective functions or very large sample sizes?  Are there variance reduction techniques or other strategies we can employ?
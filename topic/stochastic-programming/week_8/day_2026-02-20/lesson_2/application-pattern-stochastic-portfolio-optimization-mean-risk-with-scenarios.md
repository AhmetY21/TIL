---
title: "Application Pattern: Stochastic Portfolio Optimization (mean-risk with scenarios)"
date: "2026-02-20"
week: 8
lesson: 2
slug: "application-pattern-stochastic-portfolio-optimization-mean-risk-with-scenarios"
---

# Topic: Application Pattern: Stochastic Portfolio Optimization (mean-risk with scenarios)

## 1) Formal definition (what is it, and how can we use it?)

Stochastic Portfolio Optimization (SPO), specifically using a mean-risk approach with scenarios, aims to find an optimal portfolio allocation that balances expected return (mean) against risk, where the uncertainty in future asset returns is represented by a set of discrete scenarios.

**What is it?**

*   **Portfolio Optimization:** Determining the optimal allocation of capital across a set of assets to achieve a specific investment objective.
*   **Mean-Risk Framework:** The objective is to maximize expected return while minimizing risk. Risk is usually quantified by a measure of dispersion around the mean, such as variance, standard deviation, or more sophisticated risk measures like Value-at-Risk (VaR) or Conditional Value-at-Risk (CVaR).
*   **Stochastic Programming:**  A framework for optimization problems where some parameters are uncertain. Instead of assuming deterministic values, these parameters are represented by probability distributions.
*   **Scenarios:** Discrete representations of possible future outcomes. Each scenario specifies the return of each asset in that particular future "world." Each scenario is also assigned a probability representing how likely that scenario is to occur. The scenarios collectively approximate the probability distribution of asset returns.

**How can we use it?**

The mean-risk SPO with scenarios can be formulated as a mathematical optimization problem, typically a linear program (LP) or a mixed-integer linear program (MILP) depending on the complexity of the constraints and objective. The general idea is:

*   **Variables:** The decision variables are the portfolio weights (i.e., the proportion of capital allocated to each asset).
*   **Objective Function:** A combination of the expected portfolio return (averaged across all scenarios) and a risk measure. The objective is usually to maximize a risk-adjusted return (e.g., maximize expected return minus a risk penalty).
*   **Constraints:**
    *   **Budget Constraint:** The portfolio weights must sum to 1 (or potentially less if short-selling is not allowed).
    *   **Non-negativity Constraint:** If short-selling is not allowed, portfolio weights must be non-negative.
    *   **Other Constraints:** Can include investment limits on specific assets, sector diversification constraints, transaction cost constraints, or any other constraints reflecting investment policies or regulations.
*   **Solving:** The optimization problem is solved using standard optimization solvers (e.g., Gurobi, CPLEX, SciPy). The output is the optimal portfolio weights that balance expected return and risk according to the defined scenarios and constraints.

## 2) Application scenario

**Scenario:**  An investment manager wants to create an optimal portfolio using five different asset classes: US Stocks, International Stocks, Bonds, Real Estate, and Commodities. They have historical data and expert opinions, which are used to generate 100 plausible future scenarios for the returns of each asset class. Each scenario is equally likely. The manager wants to maximize the expected return of the portfolio while keeping the standard deviation of the portfolio return below a certain threshold (risk aversion). They also have a constraint that no single asset class can have more than 40% of the portfolio's capital allocated to it. Short-selling is allowed.

**Details:**

*   **Assets:** `assets = ['US Stocks', 'International Stocks', 'Bonds', 'Real Estate', 'Commodities']`
*   **Scenarios:** 100 scenarios, each with an equal probability of 1/100.  Each scenario contains a return for each asset. For example, `scenario_returns[scenario_number][asset_index]` would give the return of that asset in that specific scenario.
*   **Objective:** Maximize Expected Return - Lambda * (Standard Deviation of Returns), where Lambda is the risk aversion coefficient.
*   **Constraint:** Portfolio weights sum to 1.  Each asset class must have a weight <= 0.4.

## 3) Python method (if possible)

```python
import numpy as np
import scipy.optimize as optimize

def stochastic_portfolio_optimization(expected_returns, covariance_matrix, risk_aversion, max_weight):
    """
    Optimizes a portfolio using a mean-variance approach with a risk aversion parameter.

    Args:
        expected_returns (np.ndarray): Array of expected returns for each asset.
        covariance_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk aversion parameter (higher = more risk averse).
        max_weight (float): Maximum allowed weight for any single asset.

    Returns:
        np.ndarray: Optimal portfolio weights.
    """

    num_assets = len(expected_returns)

    # Define the objective function (negative Sharpe ratio)
    def objective(weights, expected_returns, covariance_matrix, risk_aversion):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return -(portfolio_return - risk_aversion * portfolio_volatility)

    # Define the constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights sum to 1
    bounds = [(None, max_weight)] * num_assets  # Individual asset weight constraints
    # Add non-negativity constraints

    # Initial guess (equal allocation)
    initial_weights = np.array([1/num_assets] * num_assets)

    # Optimization
    result = optimize.minimize(objective, initial_weights,
                               args=(expected_returns, covariance_matrix, risk_aversion),
                               method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        print("Optimization failed:", result.message)
        return None


# Example usage (replace with your actual data)
#Generating dummy data
num_assets = 5
num_scenarios = 100

expected_returns = np.random.rand(num_assets) * 0.1 + 0.05  # Expected returns between 5% and 15%
covariance_matrix = np.random.rand(num_assets, num_assets)
covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T) #making it positive semi definite
covariance_matrix = covariance_matrix / np.max(covariance_matrix) * 0.001  # Scaling it to more realistic values.
risk_aversion = 0.5
max_weight = 0.4
#portfolio_returns = np.random.rand(num_scenarios,num_assets)

optimal_weights = stochastic_portfolio_optimization(expected_returns, covariance_matrix, risk_aversion, max_weight)

if optimal_weights is not None:
    print("Optimal Portfolio Weights:", optimal_weights)
    print("Sum of Weights:", np.sum(optimal_weights)) # should be near 1
else:
  print("Optimization failed.")
```

**Explanation:**

1.  **`stochastic_portfolio_optimization` function:**
    *   Takes expected returns, a covariance matrix, risk aversion, and max weight for an asset as input.
    *   Defines the objective function `objective`:  This minimizes the negative Sharpe ratio, which is equivalent to maximizing risk-adjusted return.
    *   Defines constraints:  Budget constraint (weights sum to 1), and bounds constraint for maximum single asset weight.
    *   Uses `scipy.optimize.minimize` to solve the optimization problem with the SLSQP solver.
    *   Returns the optimal portfolio weights if the optimization is successful.

2.  **Example Usage:**
    *   Creates dummy data for expected returns, and a covariance matrix. It is vital for the covariance matrix to be positive semi-definite, and the matrix generation ensures this.
    *   Sets a risk aversion level.
    *   Calls the `stochastic_portfolio_optimization` function with the data.
    *   Prints the optimal portfolio weights.

**Important Considerations:**

*   **Scenario Generation:** The quality of the scenarios is crucial. They should accurately reflect the potential future outcomes and their associated probabilities. Common methods include historical simulation, Monte Carlo simulation, and bootstrapping.
*   **Risk Measure:**  The choice of risk measure significantly impacts the portfolio allocation. Standard deviation is easy to use but may not be suitable for non-normal return distributions. CVaR is a popular alternative that focuses on the tail of the distribution.
*   **Solver:**  The choice of optimization solver can affect the solution time and accuracy. Linear programs can be solved efficiently by LP solvers. More complex problems (e.g., with non-linear constraints or integer variables) may require more sophisticated solvers like MILP solvers.
*   **Transaction Costs:**  The model should ideally account for transaction costs incurred when rebalancing the portfolio.
*   **Robustness:** Scenario-based optimization can be sensitive to the specific scenarios used. Consider techniques to improve the robustness of the solution, such as scenario aggregation or robust optimization. The data used should also be reflective of the real world conditions of the asset in question.

## 4) Follow-up question

How would the formulation and code change if, instead of using standard deviation as the risk measure, we wanted to use Conditional Value-at-Risk (CVaR) at a 95% confidence level? How would this affect the computational complexity of the problem?
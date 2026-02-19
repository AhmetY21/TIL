---
title: "Evaluation Under Uncertainty: out-of-sample testing and policy evaluation"
date: "2026-02-19"
week: 8
lesson: 5
slug: "evaluation-under-uncertainty-out-of-sample-testing-and-policy-evaluation"
---

# Topic: Evaluation Under Uncertainty: out-of-sample testing and policy evaluation

## 1) Formal definition (what is it, and how can we use it?)

**Evaluation Under Uncertainty:** This refers to the process of assessing the performance of a decision-making policy or solution in a stochastic environment *after* it has been developed, and *before* its real-world deployment.  In stochastic programming, solutions are often optimized based on a sample of possible future scenarios.  The aim of evaluation under uncertainty is to determine how well the solution will generalize to scenarios *not* included in the training sample (the sample used for optimization). It's a crucial step to avoid overfitting and ensure robustness.

**Out-of-sample testing:** This involves applying the solution obtained from the training set to a completely new, independent set of scenarios (the "out-of-sample" or "test" set).  The performance on the out-of-sample set provides an estimate of the *true* performance of the policy in a real-world environment.  Common metrics include:

*   **Expected Value (EV):**  The average objective function value achieved by the policy across the out-of-sample scenarios. This gives a sense of the central tendency of performance.
*   **Value-at-Risk (VaR):**  The worst-case performance that will be exceeded with a specified probability (e.g., VaR at 95% confidence means that 95% of the time, the policy will perform better than this value).  This helps quantify downside risk.
*   **Conditional Value-at-Risk (CVaR):** The average performance in the worst-case scenarios beyond a specified probability level (e.g., CVaR at 95% is the average performance of the worst 5%).  This is a more robust measure of downside risk than VaR.
*   **Regret:**  The difference between the optimal decision made *after* knowing the realized scenario and the decision made *before* knowing the scenario. This measures the cost of making a decision under uncertainty.
*   **Stability:** Evaluates how much the decisions change given slight variations in the scenarios or the model parameters.

**Policy evaluation:** After calculating performance metrics on the out-of-sample data, you analyze these results to determine if the policy is acceptable or if further adjustments are required. This may involve:

*   **Parameter tuning:** Adjusting parameters of the optimization model or the solution process.
*   **Regularization:** Adding constraints or penalties to the optimization model to reduce overfitting and improve generalization.
*   **Robust optimization techniques:** Using techniques to explicitly account for uncertainty in the optimization process.
*   **Scenario generation improvement:** Improving the quality and representativeness of the generated scenarios.

We use out-of-sample testing and policy evaluation to:

*   **Estimate the true performance** of the stochastic programming model in a real-world setting.
*   **Identify potential weaknesses** of the model and guide improvements.
*   **Compare different policies or solution approaches** under realistic conditions.
*   **Quantify the risk** associated with using the model for decision-making.

## 2) Application scenario

Consider a power grid operator aiming to minimize operational costs while ensuring reliable electricity supply. The stochasticity comes from fluctuating renewable energy generation (solar, wind) and uncertain electricity demand.

1.  **Model Development:** A stochastic programming model is built to optimize power dispatch decisions, considering various possible scenarios of renewable generation and demand.  The model determines the optimal generation levels of different power plants (coal, gas, hydro, etc.) to meet demand at minimum cost. This optimization is performed on a training set of historical weather and demand data.

2.  **Out-of-Sample Testing:** After solving the optimization model using the training data, we need to test its performance. This involves:

    *   Generating a completely new set of scenarios of renewable generation and demand. This could be from more recent data, or a separate simulation.
    *   Using the dispatch decisions (policy) obtained from the training data to simulate the grid operation under each of these new scenarios.
    *   Calculating the operational costs for each scenario in the out-of-sample set. This will use the generation dispatch policy already learned.
    *   Also, in each scenario, check for violations of reliability constraints (e.g., voltage limits, transmission line limits).

3.  **Policy Evaluation:** Analyze the results from the out-of-sample testing.

    *   Calculate the average operational cost across all the out-of-sample scenarios (EV).
    *   Determine the 95% VaR of operational costs. This tells us the cost we might expect in a relatively bad scenario.
    *   Count the number of scenarios where reliability constraints are violated.
    *   Compare the out-of-sample performance with the performance on the training set.  If the out-of-sample performance is significantly worse, the model is likely overfitted.
    *   Based on this analysis, the operator may decide to adjust the model, improve the scenario generation process, or implement more robust dispatch strategies. For example, they might increase the amount of reserve generation or add constraints to ensure that the system remains stable even in the face of large fluctuations in renewable energy.

## 3) Python method (if possible)

```python
import numpy as np
import pandas as pd

def evaluate_policy(policy_solution, out_of_sample_scenarios, cost_function, constraint_function):
  """
  Evaluates a decision-making policy on an out-of-sample set of scenarios.

  Args:
      policy_solution:  A function (or set of parameters) representing the decision-making policy.
                       It takes a scenario as input and returns the decision (e.g., generation levels).
      out_of_sample_scenarios: A list or array of scenario data (e.g., weather data, demand data). Each element represents a single scenario.
      cost_function: A function that takes a scenario and a decision as input and returns the cost associated with that decision under that scenario.
      constraint_function: A function that takes a scenario and a decision and returns a boolean indicating whether the constraints are satisfied.

  Returns:
      A dictionary containing the performance metrics:
          - 'EV': Expected value (average cost).
          - 'VaR_95': Value-at-Risk at 95% confidence level.
          - 'CVaR_95': Conditional Value-at-Risk at 95% confidence level.
          - 'violation_rate': Proportion of scenarios where constraints are violated.
  """

  costs = []
  constraint_violations = 0

  for scenario in out_of_sample_scenarios:
    decision = policy_solution(scenario)  # Apply the policy to the scenario
    cost = cost_function(scenario, decision)  # Calculate the cost
    costs.append(cost)

    if not constraint_function(scenario, decision):
      constraint_violations += 1

  costs = np.array(costs)
  num_scenarios = len(out_of_sample_scenarios)

  EV = np.mean(costs)
  VaR_95 = np.percentile(costs, 95)
  CVaR_95 = np.mean(costs[costs >= VaR_95])  # Average of the worst 5%
  violation_rate = constraint_violations / num_scenarios

  return {
      'EV': EV,
      'VaR_95': VaR_95,
      'CVaR_95': CVaR_95,
      'violation_rate': violation_rate
  }


# Example Usage (Illustrative)

# Assume we have a trained policy that sets a single decision variable 'x'
def simple_policy(scenario):
  """A very basic example policy."""
  # The decision x is simply based on the scenario's value
  return scenario['value'] * 0.5 # example calculation

# Assume our out-of-sample scenarios are stored in a DataFrame
# with a column 'value'

# Dummy Data
num_scenarios = 100
out_of_sample_scenarios = pd.DataFrame({'value': np.random.normal(0, 1, num_scenarios)})
out_of_sample_scenarios = out_of_sample_scenarios.to_dict('records') #convert to list of dicts for easy access.

def simple_cost_function(scenario, decision):
  """Example cost function."""
  return (decision - scenario['value'])**2 # cost is the square difference

def simple_constraint_function(scenario, decision):
  """Example constraint function."""
  return decision >= 0  # Decision must be non-negative

# Evaluate the policy
results = evaluate_policy(
    policy_solution=simple_policy,
    out_of_sample_scenarios=out_of_sample_scenarios,
    cost_function=simple_cost_function,
    constraint_function=simple_constraint_function
)

print(results)
```

## 4) Follow-up question

How does the size and representativeness of the out-of-sample set affect the reliability of the evaluation process, and what techniques can be used to improve the quality of the out-of-sample evaluation, especially when real-world data is scarce?
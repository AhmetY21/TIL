---
title: "Evaluation Under Uncertainty: out-of-sample testing and policy evaluation"
date: "2026-02-19"
week: 8
lesson: 4
slug: "evaluation-under-uncertainty-out-of-sample-testing-and-policy-evaluation"
---

# Topic: Evaluation Under Uncertainty: out-of-sample testing and policy evaluation

## 1) Formal definition (what is it, and how can we use it?)

In stochastic programming, "evaluation under uncertainty" focuses on assessing the performance and robustness of a decision policy when faced with uncertain future scenarios. A key component of this is "out-of-sample testing" and "policy evaluation".

*   **Out-of-sample testing** refers to evaluating the performance of a decision policy on data that was *not* used to train or develop that policy. This is crucial because the in-sample performance (performance on the training data) can be overly optimistic due to overfitting. By testing on unseen data (out-of-sample data), we get a more realistic estimate of how the policy will perform in practice.  This "out-of-sample" data is drawn from the same (or a representative) distribution as the data on which the policy is eventually deployed. Out-of-sample tests typically involve generating a large number of scenarios (or realizations) of the uncertain parameters and evaluating the decision policy under each scenario. The results are then aggregated (e.g., averaging the objective function values) to obtain an overall performance metric.

*   **Policy evaluation** involves estimating the value or performance of a particular decision policy under uncertainty.  This means characterizing the expected outcome (e.g., cost, profit, or other objective function value) when following the specific decision rules prescribed by that policy, given the distribution of uncertain parameters.  Policy evaluation often combines simulation (generating scenarios) with deterministic optimization (solving the problem for each scenario *given* the decision made by the policy). Policy evaluation methods may be used to compare different decision policies or to assess the sensitivity of a particular policy to changes in the underlying uncertainties.

We use these tools to:

*   **Assess policy robustness:** Determine how well the policy performs under various uncertain scenarios.
*   **Compare different policies:** Evaluate and rank multiple policies to select the best one.
*   **Quantify risk:** Measure the potential downside risks associated with a policy, such as the probability of poor performance or constraint violations.
*   **Improve policy design:** Identify weaknesses in a policy and guide improvements by understanding its behavior in different scenarios.
*   **Build confidence:** Provide stakeholders with confidence in the policy's performance by demonstrating its effectiveness across a range of plausible future outcomes.

## 2) Application scenario

Consider a supply chain planning problem where demand for a product is uncertain.  A company needs to decide how much inventory to order each period.

*   **Decision policy:**  A simple policy might be a base-stock policy: "Order enough inventory to bring the total inventory level up to a target level of S." Here, 'S' is the decision variable.

*   **Uncertainty:**  The demand for the product in each period is uncertain and follows a probability distribution (e.g., a normal distribution with a known mean and variance).

*   **Out-of-sample testing:** To evaluate this policy, we would:

    1.  Generate a large number of demand scenarios (e.g., 1000 scenarios) for a future planning horizon (e.g., 12 months). Each scenario represents a possible realization of demand in each month.
    2.  For *each* demand scenario:
        *   Simulate the supply chain operation using the base-stock policy with a specific value of S.  This involves tracking inventory levels, placing orders, fulfilling demand, and incurring costs such as holding costs (for excess inventory) and shortage costs (for unmet demand).
    3.  Calculate the average total cost (across all scenarios) for each value of S.

    We would then repeat this process for different values of S to find the value of S that minimizes the average total cost in the out-of-sample data. This gives us a robust estimate of the optimal base-stock level.

*   **Policy Evaluation:** This is almost identical to the out-of-sample testing, but the *purpose* is slightly different.  Instead of optimizing the policy (choosing S), the goal is to *evaluate* a specific, pre-defined value of S. The steps are the same, but the focus is on reporting statistics (e.g., average cost, standard deviation of cost, worst-case cost) for the chosen value of S.

## 3) Python method (if possible)

```python
import numpy as np

def evaluate_base_stock_policy(S, demand_scenarios, holding_cost, shortage_cost):
  """
  Evaluates a base-stock inventory policy given demand scenarios.

  Args:
    S: Base-stock level (target inventory level).
    demand_scenarios: A numpy array of shape (num_scenarios, num_periods)
                      representing demand in each period for each scenario.
    holding_cost: Cost of holding one unit of inventory for one period.
    shortage_cost: Cost of having a shortage of one unit of demand.

  Returns:
    A tuple: (average_total_cost, costs_per_scenario) where
             average_total_cost is the average total cost across all scenarios, and
             costs_per_scenario is a numpy array containing total costs for each scenario.
  """

  num_scenarios, num_periods = demand_scenarios.shape
  costs_per_scenario = np.zeros(num_scenarios)

  for scenario_idx in range(num_scenarios):
    inventory = 0  # Initial inventory
    total_cost = 0
    for period in range(num_periods):
      demand = demand_scenarios[scenario_idx, period]

      # Order to reach base-stock level
      order = max(0, S - inventory)
      inventory += order

      # Meet demand
      inventory -= demand
      if inventory < 0:
        total_cost += -inventory * shortage_cost # Shortage cost
        inventory = 0  # inventory cannot be negative, assume backlog
      else:
        total_cost += inventory * holding_cost  # Holding cost

    costs_per_scenario[scenario_idx] = total_cost

  average_total_cost = np.mean(costs_per_scenario)
  return average_total_cost, costs_per_scenario

# Example usage:
S = 100  # Base-stock level
num_scenarios = 1000
num_periods = 12
demand_scenarios = np.random.normal(loc=80, scale=20, size=(num_scenarios, num_periods)) # Simulate demand

holding_cost = 1
shortage_cost = 10

average_cost, scenario_costs = evaluate_base_stock_policy(S, demand_scenarios, holding_cost, shortage_cost)

print(f"Average total cost: {average_cost}")

# You can then analyze the scenario_costs to understand the distribution of costs
# e.g., calculate the Value at Risk (VaR) or Conditional Value at Risk (CVaR).

```

This Python code simulates a simple inventory system and calculates the average total cost under a given base-stock policy. The `evaluate_base_stock_policy` function takes the base-stock level, demand scenarios, holding cost, and shortage cost as input and returns the average total cost across all scenarios, as well as an array of the costs for each individual scenario. The demand scenarios are generated using NumPy's random number generator. This can be adapted for other types of demand distributions or other stochastic programming problems. Note that this is a *simulation*-based policy evaluation, and it does not involve optimizing the base stock level S. If one wished to *optimize* S via simulation, one would need to repeatedly call `evaluate_base_stock_policy` for different values of S and select the S value that minimizes the returned `average_total_cost`.

## 4) Follow-up question

How can variance reduction techniques (e.g., common random numbers, antithetic variates) be applied to improve the efficiency of out-of-sample testing and policy evaluation in stochastic programming, and what are the practical considerations when choosing a specific variance reduction technique?
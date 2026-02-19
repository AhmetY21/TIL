---
title: "Evaluation Under Uncertainty: out-of-sample testing and policy evaluation"
date: "2026-02-19"
week: 8
lesson: 2
slug: "evaluation-under-uncertainty-out-of-sample-testing-and-policy-evaluation"
---

# Topic: Evaluation Under Uncertainty: out-of-sample testing and policy evaluation

## 1) Formal definition (what is it, and how can we use it?)

**Evaluation under uncertainty** in the context of stochastic programming refers to assessing the performance and robustness of a decision policy (or a solution obtained through stochastic programming) when exposed to unseen, or out-of-sample, scenarios. It's a critical step after obtaining a solution from a stochastic programming model. It aims to understand how the solution will behave in "real-world" conditions which are often different from the data used during the model training or optimization phase.

**Out-of-sample testing** specifically involves evaluating the decision policy on a new set of scenarios that were *not* used to train or optimize the stochastic programming model. This is akin to testing a machine learning model on a held-out test set. The idea is to simulate the situation where the decision policy is deployed in the future and encounters conditions different from those experienced during its creation.

**Policy evaluation** extends the concept of out-of-sample testing. While out-of-sample testing focuses on direct performance metrics like cost or profit, policy evaluation aims to understand the policy's overall effectiveness, risk profile, and sensitivity to different scenarios. It may involve:

*   **Calculating expected value of perfect information (EVPI):** This metric quantifies the benefit of knowing the uncertain parameters with certainty before making a decision. It provides an upper bound on how much we should be willing to pay for better information.
*   **Value of the stochastic solution (VSS):** This metric measures the improvement in performance (e.g., cost reduction, profit increase) achieved by using a stochastic programming approach compared to a deterministic approach that only considers expected values for uncertain parameters.
*   **Scenario analysis:** Examining the policy's performance across different, potentially extreme, scenarios to identify vulnerabilities.
*   **Risk analysis:** Assessing the probability of undesirable outcomes (e.g., losses exceeding a certain threshold) under different scenarios.

**How can we use it?**

*   **Model Validation:** Verify if the model accurately represents the real-world problem by checking if the solution performs reasonably well out-of-sample.
*   **Policy Comparison:** Compare the performance of different solution approaches obtained from stochastic programming or other optimization methods under uncertainty.
*   **Risk Management:** Identify and quantify the risks associated with implementing a specific decision policy, allowing for the development of mitigation strategies.
*   **Stakeholder Communication:** Provide realistic performance expectations to stakeholders based on the policy's behavior across a range of plausible scenarios.
*   **Improve the Model:** Out-of-sample performance can highlight areas where the model needs refinement, such as incorporating additional constraints or using a more accurate representation of uncertainty.

## 2) Application scenario

Consider a supply chain network design problem. A company needs to decide where to build warehouses and how much inventory to hold at each location to meet customer demand. Demand is uncertain and is modeled using a set of scenarios. A stochastic programming model is used to find the optimal warehouse locations and inventory levels.

After solving the stochastic programming model, we need to evaluate its performance out-of-sample. We generate a new set of demand scenarios that were *not* used in the original optimization. These new scenarios represent potential future market conditions.

We then simulate the operation of the supply chain under these new demand scenarios, using the warehouse locations and inventory levels determined by the stochastic programming model. We track metrics such as total cost (including transportation, warehousing, and shortage costs), service level (percentage of demand met on time), and the number of stockouts.

By analyzing these metrics across the out-of-sample scenarios, we can assess the robustness of the solution. For example, we might find that the solution performs well on average but is highly vulnerable to extreme demand fluctuations in certain regions. This information can then be used to refine the supply chain design (e.g., by adding more safety stock or diversifying sourcing) and provide a more realistic estimate of the supply chain's performance under real-world conditions.

## 3) Python method (if possible)

```python
import numpy as np

# Assume we have a function to simulate the supply chain
# and calculate the total cost for a given scenario
def simulate_supply_chain(warehouse_locations, inventory_levels, demand_scenario):
  """
  Simulates the supply chain and returns the total cost.

  Args:
    warehouse_locations: A list of warehouse locations.
    inventory_levels: A list of inventory levels for each warehouse.
    demand_scenario: A dictionary representing demand at each location.

  Returns:
    The total cost of the supply chain.
  """
  # Replace with your actual simulation logic
  transportation_cost = np.sum(np.random.rand(len(warehouse_locations))) * np.sum(list(demand_scenario.values()))
  warehousing_cost = np.sum(inventory_levels) * 0.1
  shortage_cost = max(0, sum(demand_scenario.values()) - sum(inventory_levels)) * 2
  total_cost = transportation_cost + warehousing_cost + shortage_cost
  return total_cost

# Example usage:
# 1. Obtain a solution (warehouse locations, inventory levels) from your stochastic programming model.
optimal_warehouse_locations = [1, 2, 3] # Example warehouse locations
optimal_inventory_levels = [100, 150, 200] # Example inventory levels

# 2. Generate out-of-sample scenarios.
num_scenarios = 100
out_of_sample_scenarios = []
for _ in range(num_scenarios):
  demand = {
      "location_1": np.random.randint(50, 150),
      "location_2": np.random.randint(75, 225),
      "location_3": np.random.randint(100, 300)
  }
  out_of_sample_scenarios.append(demand)

# 3. Evaluate the solution on the out-of-sample scenarios.
costs = []
for scenario in out_of_sample_scenarios:
  cost = simulate_supply_chain(optimal_warehouse_locations, optimal_inventory_levels, scenario)
  costs.append(cost)

# 4. Analyze the results.
average_cost = np.mean(costs)
std_dev_cost = np.std(costs)
max_cost = np.max(costs)
min_cost = np.min(costs)

print(f"Average cost: {average_cost}")
print(f"Standard deviation of cost: {std_dev_cost}")
print(f"Maximum cost: {max_cost}")
print(f"Minimum cost: {min_cost}")

# You can also analyze the distribution of costs using histograms or other visualizations.
import matplotlib.pyplot as plt

plt.hist(costs, bins=20)
plt.xlabel("Total Cost")
plt.ylabel("Frequency")
plt.title("Distribution of Costs in Out-of-Sample Scenarios")
plt.show()
```

**Explanation:**

1.  **`simulate_supply_chain` function:**  This function simulates the supply chain given the warehouse locations, inventory levels, and a specific demand scenario. This is a placeholder. You'll need to replace this with the actual simulation logic for your specific problem.  It calculates the total cost, considering transportation, warehousing, and potential shortage costs.  Crucially, the specifics of your simulation will drastically influence your result.

2.  **Generating out-of-sample scenarios:** A loop generates `num_scenarios` demand scenarios.  The `demand` dictionary represents the demand at different locations. The values are randomly generated within a range. You should use a distribution and parameters that are appropriate for your problem (e.g., normal distribution, historical data).

3.  **Evaluating the solution:** The code iterates through the out-of-sample scenarios and calls the `simulate_supply_chain` function to calculate the cost for each scenario.

4.  **Analyzing the results:** The code calculates the average cost, standard deviation, maximum cost, and minimum cost. It also generates a histogram of the costs using `matplotlib`.  These statistics provide insights into the performance and robustness of the solution.  You can further calculate risk measures like Value at Risk (VaR) or Conditional Value at Risk (CVaR).

**Important notes:**

*   The `simulate_supply_chain` function is a simplified example.  A real-world simulation would be much more complex and would involve considering various factors, such as transportation times, capacity constraints, and service level requirements.
*   The choice of the distribution and parameters for generating the out-of-sample scenarios is crucial.  You should use a distribution that accurately represents the uncertainty in your problem and parameters that are based on historical data or expert judgment.
*   The number of out-of-sample scenarios should be large enough to provide a statistically significant estimate of the solution's performance.

## 4) Follow-up question

How does the choice of the scenario generation method (e.g., Monte Carlo sampling, quasi-Monte Carlo sampling, historical data resampling) affect the validity and reliability of out-of-sample testing and policy evaluation in stochastic programming, and what are the trade-offs involved in choosing one method over another? Consider also the impact of *correlated* uncertainties.
---
title: "Evaluation Under Uncertainty: out-of-sample testing and policy evaluation"
date: "2026-02-19"
week: 8
lesson: 3
slug: "evaluation-under-uncertainty-out-of-sample-testing-and-policy-evaluation"
---

# Topic: Evaluation Under Uncertainty: out-of-sample testing and policy evaluation

## 1) Formal definition (what is it, and how can we use it?)

Evaluation under uncertainty, particularly through out-of-sample testing and policy evaluation, is the process of assessing the performance and robustness of a stochastic programming model or decision rule (policy) when faced with unseen, uncertain data.  It's a crucial step after developing and training a model to ensure it generalizes well and performs as expected in real-world scenarios.

*   **Out-of-sample testing:**  This involves evaluating the model's performance on a dataset *not* used during the model's development (e.g., training or in-sample validation). This provides a more realistic estimate of how the model will perform on future, unseen data.  We use the trained stochastic program to make decisions using these new realizations of the random parameters, and measure the resulting objective value.

*   **Policy Evaluation:** Once a stochastic program yields a proposed *policy* (e.g., a set of decision rules or a closed-form solution) then *policy evaluation* seeks to quantify the performance of this policy against different possible future states. It involves simulating the consequences of applying the policy under various scenarios generated from the assumed uncertainty distributions. This may take the form of Monte Carlo simulation where we generate a very large number of scenarios, apply the policy, and then average the results.

**How we use it:**

*   **Model Validation:** Out-of-sample testing validates whether the model captures the underlying relationships in the data and avoids overfitting to the training data. This informs whether the selected model structure and the assumed uncertainty set are appropriate.
*   **Performance Assessment:** It provides a more realistic estimate of the expected performance of the model in real-world applications, measuring things like average cost, worst-case performance, and risk metrics.
*   **Policy Comparison:**  Allows us to compare different decision policies (e.g., different parameter settings of the stochastic program) to determine which performs best under uncertainty.
*   **Risk Management:** Helps identify potential vulnerabilities of the model and assess the risk associated with different decision strategies. It helps to estimate the chance of falling below certain performance levels.
*   **Decision Support:** Informs decision-makers about the expected outcomes and potential risks associated with implementing the model's recommendations.

## 2) Application scenario

Consider a supply chain planning problem where a company needs to decide on the inventory levels of several products to meet uncertain demand.

**In-Sample Optimization:** The company builds a stochastic program that takes into account historical demand data and incorporates uncertainty through probability distributions. They optimize the inventory levels based on this in-sample data.

**Out-of-Sample Testing:** To evaluate the robustness of their model, they generate *new* demand scenarios from the same (or similar) probability distributions. They then use the inventory levels determined by the in-sample optimization and *simulate* the performance of the supply chain under these new scenarios.  They calculate the resulting profit or cost (taking into account inventory holding costs, shortage penalties, etc.) for each scenario and average them to get an estimate of the expected performance.

**Policy Evaluation:** Suppose the solution to the stochastic program yields a policy like: "Order enough to bring your inventory level to three times the standard deviation above the mean demand, regardless of the starting inventory." We can then generate, say 10,000 demand realizations, and for each realization, apply the *policy* to calculate an order level, and then simulate the system. The average objective value (profit or cost) calculated across these 10,000 scenarios provides an estimate of the expected performance of the policy.

If the out-of-sample performance is significantly worse than the in-sample performance, it suggests the model is overfitting to the training data or that the chosen uncertainty set is too small (and that the true uncertainty is larger). This might prompt the company to refine the model or consider more robust optimization approaches. Alternatively, policy evaluation may reveal that the chosen policy is not robust, and a new policy (e.g., requiring different inventory targets under different demand conditions) should be explored.

## 3) Python method (if possible)

```python
import numpy as np
import random

def evaluate_policy(policy, demand_distributions, num_scenarios):
    """
    Evaluates a given inventory policy under uncertain demand.

    Args:
        policy (callable): A function that takes demand information (mean, std_dev) and returns an order quantity.
        demand_distributions (dict): A dictionary where keys are products and values are tuples (mean_demand, std_dev_demand).
        num_scenarios (int): The number of demand scenarios to simulate.

    Returns:
        float: The average profit across all scenarios.
    """
    total_profit = 0
    for _ in range(num_scenarios):
        # Generate a scenario of random demands for each product
        demands = {}
        for product, (mean, std_dev) in demand_distributions.items():
            demands[product] = max(0, np.random.normal(mean, std_dev)) # Ensure demand is non-negative

        # Determine order quantities based on the policy
        order_quantities = {}
        for product, (mean, std_dev) in demand_distributions.items():
             order_quantities[product] = policy(mean, std_dev) # Assuming the policy does not consider current inventory

        # Simulate the outcome: Calculate profit based on demand and order quantities.
        # This is a simplified example; you'd need a more complex profit calculation
        # incorporating costs like holding costs, shortage costs, etc.
        profit = 0
        for product in demand_distributions:
            sales = min(order_quantities[product], demands[product])
            profit += sales * 10 - order_quantities[product]*2 #Simplified example: selling price 10, order cost 2

        total_profit += profit

    return total_profit / num_scenarios


# Example Usage:

# Define a simple policy: Order to reach a target inventory level.  This function returns the ORDER quantity to reach the target.
def simple_policy(mean_demand, std_dev_demand):
    target_inventory = mean_demand + 2 * std_dev_demand # Target inventory 2 standard deviations above mean
    return target_inventory # The order quantity


# Define demand distributions for two products: A and B
demand_distributions = {
    "A": (50, 10),  # Mean demand 50, standard deviation 10
    "B": (80, 15),  # Mean demand 80, standard deviation 15
}

# Evaluate the policy
num_scenarios = 1000
average_profit = evaluate_policy(simple_policy, demand_distributions, num_scenarios)

print(f"Average Profit under Uncertainty: {average_profit}")
```

This Python code provides a basic framework for policy evaluation using Monte Carlo simulation.  Key considerations:

*   The `evaluate_policy` function simulates the supply chain under multiple demand scenarios.
*   The `simple_policy` represents an example decision rule. This would be replaced with the actual decision rule resulting from the stochastic programming model.
*   The profit calculation is simplified. A more realistic model would include various cost components.
*   This example focuses on policy evaluation (estimating performance given a fixed policy). Out-of-sample *testing* in the pure sense would involve running the *entire* optimization process on a separate set of data, and then *comparing* the decision variables that result. This is not shown here for brevity.

## 4) Follow-up question

How can variance reduction techniques (e.g., control variates, antithetic variates) be applied to improve the efficiency and accuracy of policy evaluation in stochastic programming, especially when dealing with computationally expensive simulations or high-dimensional uncertainty?
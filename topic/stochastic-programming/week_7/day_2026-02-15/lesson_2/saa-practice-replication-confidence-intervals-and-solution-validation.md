---
title: "SAA Practice: replication, confidence intervals, and solution validation"
date: "2026-02-15"
week: 7
lesson: 2
slug: "saa-practice-replication-confidence-intervals-and-solution-validation"
---

# Topic: SAA Practice: replication, confidence intervals, and solution validation

## 1) Formal definition (what is it, and how can we use it?)

Sample Average Approximation (SAA) is a method for approximating the solution to stochastic optimization problems.  These problems involve decision variables and parameters that are random variables.  Since we cannot directly solve problems with infinite possibilities, SAA approximates the true problem by replacing the probability distributions of the random parameters with a finite sample of those parameters.

* **Replication:**  SAA involves generating *multiple independent* samples of the random parameters.  Each sample produces a solution to the deterministic approximation of the stochastic problem.  Running the optimization for each sample is a *replication*. Replication is essential because the solution quality (optimality) of a single SAA run can vary significantly due to the randomness of the single sample used.

* **Confidence Intervals:** Since the SAA solution is an approximation, we need to estimate the quality of the obtained solution. Confidence intervals provide a range within which the true optimal value is likely to lie, given a specified confidence level (e.g., 95%). This involves estimating the statistical properties (mean and variance) of the estimated optimal objective values obtained across multiple replications. Narrower confidence intervals indicate more reliable solutions. Computing confidence intervals relies on statistical assumptions, often normality (which might need validation) of the average objective function values across replicates.

* **Solution Validation:**  After obtaining a candidate solution from SAA, it is crucial to validate its performance on a separate, independent test sample.  This *out-of-sample* validation provides an unbiased estimate of the true cost or reward associated with implementing the proposed decision in the real, uncertain environment.  The performance on the test sample is typically worse than on the training sample used in the SAA, due to overfitting on the specific training sample. Solution validation allows us to quantify the degree of overfitting and assess the robustness of the proposed solution. This typically involves evaluating the objective function value (or a relevant metric like risk) for the chosen solution, using the test sample. This metric should be comparable to the approximated objective function value derived from the SAA runs.

**How we can use it:**

1.  **Obtain multiple SAA solutions:** Generate several independent samples, and solve the deterministic equivalent problem for each.
2.  **Estimate the optimal objective value:** Calculate the average objective function value across all SAA solutions.
3.  **Calculate confidence intervals:** Compute a confidence interval for the true optimal objective value using the SAA solutions.
4.  **Validate the solution:** Evaluate the performance of the selected solution (e.g., the solution that yields the lowest average cost across all replications) on an independent test sample. This means fixing the decision variables to the optimal values obtained from the SAA runs and then computing the expected cost using the test data.
5.  **Compare with benchmarks:**  Compare the validation performance with benchmark solutions (e.g., a simple heuristic or the optimal solution for a deterministic version of the problem).
6.  **Iterate and refine:** If the confidence intervals are too wide or the out-of-sample performance is poor, increase the sample size or refine the optimization model.

## 2) Application scenario

Consider a supply chain management problem where a company needs to decide how much inventory to order for a product at multiple warehouses. The demand at each warehouse is uncertain.  We want to minimize the total cost, which includes ordering costs, holding costs, and shortage costs.

1.  **Stochastic Problem:** The demand at each warehouse is a random variable with a known probability distribution (e.g., Normal, Poisson).
2.  **SAA:** We can use SAA to approximate the stochastic problem by generating samples of demand scenarios.
3.  **Replication:** Generate, say, 50 independent samples of the demand scenarios. For each sample, solve the deterministic inventory optimization problem (e.g., using linear programming). This gives us 50 different inventory ordering policies (SAA solutions).
4.  **Confidence Intervals:** Compute a 95% confidence interval for the expected total cost. If the confidence interval is wide, it suggests that more samples are needed to obtain a more reliable estimate.
5.  **Solution Validation:** Select the inventory policy that yields the lowest average cost across the 50 samples.  Then, evaluate this policy on a separate, independent test sample (e.g., 1000 demand scenarios). This out-of-sample test gives us an unbiased estimate of the true cost associated with implementing the chosen inventory policy.  We compare this cost with the average cost obtained in the SAA replications, to see the difference. We could also compare this policy with a simple rule-of-thumb policy.

## 3) Python method (if possible)

```python
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

# Example stochastic problem (simplified): Single warehouse inventory control
def cost_function(x, demand):
    """
    Calculates the total cost (ordering, holding, shortage) for a given order quantity and demand.
    """
    order_quantity = x[0] # Scalar order quantity
    holding_cost = 1
    shortage_cost = 5
    ordering_cost = 2
    
    holding = max(order_quantity - demand, 0)
    shortage = max(demand - order_quantity, 0)
    
    total_cost = ordering_cost + holding_cost * holding + shortage_cost * shortage
    return total_cost


def saa_replication(num_samples, demand_mean, demand_std):
    """
    Performs SAA replication for the inventory control problem.
    """
    solutions = []
    objective_values = []

    for _ in range(num_samples):
        # Generate random demand samples
        demand_samples = np.random.normal(demand_mean, demand_std, size=100) # Simulate demand 100 times

        # Define objective function for a single sample
        def objective_function(x):
            avg_cost = np.mean([cost_function(x, d) for d in demand_samples])
            return avg_cost

        # Solve optimization problem for this sample
        result = minimize(objective_function, x0=[demand_mean], bounds=[(0, None)]) # Start at mean
        solutions.append(result.x[0])  # Optimal order quantity
        objective_values.append(result.fun) # Optimal objective value for this sample

    return solutions, objective_values


def calculate_confidence_interval(objective_values, confidence_level=0.95):
    """
    Calculates a confidence interval for the true expected cost.
    """
    n = len(objective_values)
    mean = np.mean(objective_values)
    std_err = np.std(objective_values) / np.sqrt(n)
    alpha = 1 - confidence_level
    
    # Using t-distribution for more accurate CIs with smaller sample sizes.
    t_critical = t.ppf(1 - alpha/2, n-1)
    margin_of_error = t_critical * std_err

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def validate_solution(solution, demand_mean, demand_std, num_test_samples=1000):
    """
    Validates the SAA solution on an independent test sample.
    """
    test_demand_samples = np.random.normal(demand_mean, demand_std, size=num_test_samples)
    total_cost = np.mean([cost_function([solution], d) for d in test_demand_samples])
    return total_cost


# Example usage
if __name__ == '__main__':
    num_replications = 30
    demand_mean = 50
    demand_std = 10

    # Perform SAA replications
    solutions, objective_values = saa_replication(num_replications, demand_mean, demand_std)

    # Calculate confidence interval
    lower_bound, upper_bound = calculate_confidence_interval(objective_values)
    print(f"Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

    # Select the best solution (lowest average cost)
    best_solution = solutions[np.argmin(objective_values)]
    print(f"Best Solution (Order Quantity): {best_solution:.2f}")
    
    # Validate the solution
    validation_cost = validate_solution(best_solution, demand_mean, demand_std)
    print(f"Validation Cost: {validation_cost:.2f}")
```

## 4) Follow-up question

How does the choice of the number of samples in SAA (both for training and validation) impact the accuracy of the solution and the computational cost? What strategies can be used to determine an appropriate sample size for a given problem?
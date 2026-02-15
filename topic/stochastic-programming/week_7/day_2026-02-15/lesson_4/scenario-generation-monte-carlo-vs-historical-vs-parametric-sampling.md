---
title: "Scenario Generation: Monte Carlo vs historical vs parametric sampling"
date: "2026-02-15"
week: 7
lesson: 4
slug: "scenario-generation-monte-carlo-vs-historical-vs-parametric-sampling"
---

# Topic: Scenario Generation: Monte Carlo vs historical vs parametric sampling

## 1) Formal definition (what is it, and how can we use it?)

Scenario generation in stochastic programming aims to create a discrete set of possible future outcomes (scenarios) for uncertain parameters that significantly impact a decision-making process. Instead of assuming certainty or using average values, it acknowledges the inherent randomness and potential variability of real-world variables like demand, prices, weather, or technology costs. These scenarios are then used to optimize decisions while considering the potential range of outcomes and their associated probabilities.

**What is it?**

Scenario generation involves creating a set of discrete scenarios, each with an associated probability, that represent possible future realizations of uncertain parameters. The goal is to approximate the true underlying probability distribution of the uncertain parameters in a way that is computationally tractable for optimization.

**How can we use it?**

Scenarios are used within stochastic programming models to:

*   **Account for uncertainty:** By explicitly incorporating multiple possible futures, stochastic programming can find solutions that are robust and perform well across different scenarios.
*   **Risk management:** Stochastic programming can assess the impact of different scenarios on the objective function and identify potential risks. This allows for proactive measures to mitigate adverse outcomes.
*   **Decision-making under uncertainty:** The solution derived from a stochastic programming model considers the trade-offs between different scenarios, leading to more informed and robust decisions.
*   **Model Input:** The method of scenario generation is directly used as input in other stochastic programming modules, for example, in stochastic optimization models.

The main methods for scenario generation are:

*   **Monte Carlo sampling:** This approach involves simulating the underlying probability distribution of the uncertain parameters and generating a large number of scenarios.
*   **Historical sampling:** This approach uses historical data to create scenarios based on past observations.
*   **Parametric sampling:** This approach assumes a specific probability distribution for the uncertain parameters and then generates scenarios based on the parameters of that distribution.

## 2) Application scenario

**Supply Chain Planning Under Demand Uncertainty:**

Imagine a company that produces and sells a product with uncertain demand. The company needs to decide how much product to produce and stock at different distribution centers before the actual demand is realized.

*   **Monte Carlo Sampling:** If the demand process is governed by a complex simulation model, such as a consumer choice model, Monte Carlo sampling can be used to generate demand scenarios. The simulation model is run many times with different random inputs, and each run generates a different demand scenario for each distribution center.

*   **Historical Sampling:** The company has five years of historical demand data for each distribution center. Historical sampling can be used to create scenarios by resampling from this historical data. For example, each scenario could be created by randomly selecting one year of historical data for each distribution center.

*   **Parametric Sampling:** Suppose the company believes that demand for each distribution center follows a normal distribution, and they have estimated the mean and standard deviation of demand for each distribution center based on historical data. Parametric sampling can be used to generate demand scenarios by drawing random samples from the assumed normal distributions.

In this scenario, stochastic programming can then be used to determine the optimal production and inventory levels at each distribution center, taking into account the different demand scenarios and their associated probabilities. The goal is to minimize the total cost of production, inventory holding, and lost sales, while ensuring a high level of customer service.

## 3) Python method (if possible)

```python
import numpy as np
import pandas as pd

def monte_carlo_scenario_generation(num_scenarios, distribution_params):
    """
    Generates scenarios using Monte Carlo sampling for multiple uncertain variables.

    Args:
        num_scenarios: The number of scenarios to generate.
        distribution_params: A dictionary where keys are variable names and values
                             are dictionaries containing the distribution type ('normal', 'uniform', etc.)
                             and the corresponding parameters (e.g., mean and std for normal).
                             Example:
                             {
                                 'demand': {'distribution': 'normal', 'mean': 100, 'std': 20},
                                 'price': {'distribution': 'uniform', 'low': 10, 'high': 15}
                             }

    Returns:
        A pandas DataFrame where each row represents a scenario and each column represents
        an uncertain variable.
    """
    scenarios = {}
    for variable, params in distribution_params.items():
        dist_type = params['distribution']
        if dist_type == 'normal':
            scenarios[variable] = np.random.normal(loc=params['mean'], scale=params['std'], size=num_scenarios)
        elif dist_type == 'uniform':
            scenarios[variable] = np.random.uniform(low=params['low'], high=params['high'], size=num_scenarios)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    return pd.DataFrame(scenarios)


def historical_scenario_generation(num_scenarios, historical_data: pd.DataFrame, replace=True):
    """
    Generates scenarios by resampling from historical data.

    Args:
        num_scenarios: The number of scenarios to generate.
        historical_data: A pandas DataFrame containing historical data. Each row represents a historical observation.
        replace: Whether to sample with replacement (default: True).

    Returns:
        A pandas DataFrame where each row represents a scenario sampled from historical data.
    """
    # Ensure historical_data is a DataFrame
    if not isinstance(historical_data, pd.DataFrame):
        raise TypeError("historical_data must be a pandas DataFrame")

    # Ensure that there is enough data
    if num_scenarios > len(historical_data) and not replace:
      raise ValueError(f"Number of scenarios, {num_scenarios}, is greater than the number of historical data points, {len(historical_data)}, and sampling without replacement is specified.")

    #Sample historical data points using a random sample generator
    sampled_indices = np.random.choice(len(historical_data), size=num_scenarios, replace=replace)
    return historical_data.iloc[sampled_indices].copy() #important to copy so no warning is thrown if the original data is edited later

def parametric_scenario_generation(num_scenarios, distribution_parameters):
    """
    Generates scenarios using pre-defined parametric distributions for multiple variables.

    Args:
        num_scenarios: The number of scenarios to generate.
        distribution_parameters: Dictionary defining parameters for each variable.  Keys are variable names, values are tuples: (distribution_type, *distribution_params).
            Supported distribution_types are 'normal' (mean, std), 'uniform' (low, high), 'gamma' (shape, scale), 'exponential' (scale).

    Returns:
        pandas.DataFrame: A DataFrame where each row is a scenario and columns are the generated variable values.
    """

    scenarios = {}
    for variable, (dist_type, *params) in distribution_parameters.items():
        if dist_type == 'normal':
            mean, std = params
            scenarios[variable] = np.random.normal(loc=mean, scale=std, size=num_scenarios)
        elif dist_type == 'uniform':
            low, high = params
            scenarios[variable] = np.random.uniform(low=low, high=high, size=num_scenarios)
        elif dist_type == 'gamma':
            shape, scale = params
            scenarios[variable] = np.random.gamma(shape=shape, scale=scale, size=num_scenarios)
        elif dist_type == 'exponential':
            scale = params[0]
            scenarios[variable] = np.random.exponential(scale=scale, size=num_scenarios)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type} for variable {variable}")

    return pd.DataFrame(scenarios)

# Example Usage:

# Monte Carlo
distribution_params = {
    'demand': {'distribution': 'normal', 'mean': 100, 'std': 20},
    'price': {'distribution': 'uniform', 'low': 10, 'high': 15}
}
mc_scenarios = monte_carlo_scenario_generation(num_scenarios=10, distribution_params=distribution_params)
print("Monte Carlo Scenarios:\n", mc_scenarios)

# Historical
historical_data = pd.DataFrame({
    'demand': [80, 90, 110, 100, 120],
    'price': [12, 11, 13, 14, 15]
})
hist_scenarios = historical_scenario_generation(num_scenarios=10, historical_data=historical_data)
print("\nHistorical Scenarios:\n", hist_scenarios)

# Parametric
distribution_parameters = {
    'demand': ('normal', 100, 20), # Normal distribution, mean=100, std=20
    'price': ('uniform', 10, 15),   # Uniform distribution, low=10, high=15
    'lead_time': ('gamma', 2, 1)    # Gamma distribution, shape=2, scale=1
}

param_scenarios = parametric_scenario_generation(num_scenarios=10, distribution_parameters=distribution_parameters)
print("\nParametric Scenarios:\n", param_scenarios)
```

## 4) Follow-up question

How do you determine the appropriate number of scenarios to generate for a stochastic programming model to balance computational cost and solution accuracy? Are there any rules of thumb or methods to assess the quality and representativeness of the generated scenarios?
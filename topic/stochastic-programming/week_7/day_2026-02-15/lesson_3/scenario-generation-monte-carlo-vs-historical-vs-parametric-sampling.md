---
title: "Scenario Generation: Monte Carlo vs historical vs parametric sampling"
date: "2026-02-15"
week: 7
lesson: 3
slug: "scenario-generation-monte-carlo-vs-historical-vs-parametric-sampling"
---

# Topic: Scenario Generation: Monte Carlo vs historical vs parametric sampling

## 1) Formal definition (what is it, and how can we use it?)

Scenario generation, in the context of stochastic programming, is the process of creating a finite set of plausible future realizations (scenarios) of uncertain parameters.  Stochastic programming models incorporate these scenarios to make decisions that are robust across different potential future states.  Rather than treating uncertain parameters as fixed values, stochastic programming allows for variability, reflecting the inherent uncertainty in real-world problems. The quality of the scenarios directly impacts the quality of the solution. Good scenarios should:

*   **Represent the underlying uncertainty appropriately:** They should capture the range of possible values and their probabilities.
*   **Be realistic:** Avoid scenarios that are completely implausible in the real world.
*   **Be computationally manageable:**  The number of scenarios significantly affects the computational complexity of solving the stochastic program.  Too many scenarios can make the problem intractable, while too few may lead to suboptimal or unrealistic decisions.

Scenario generation techniques aim to achieve these goals using different approaches. Here's a breakdown of the three methods:

*   **Monte Carlo Sampling:** This method involves simulating the uncertain parameters based on their assumed probability distributions.  We define the distribution(s) (e.g., normal, uniform, exponential) for each uncertain parameter and then generate a large number of random samples from these distributions. Each sample represents a possible scenario. We can then use a subset of these samples, or potentially reduce the number via scenario reduction techniques, to represent the uncertainty in the stochastic programming model.

*   **Historical Sampling:** This method uses historical data to create scenarios. We collect past observations of the uncertain parameters and treat each observation as a possible scenario. This is useful when probability distributions are unknown or difficult to estimate, but sufficient historical data is available.  Variations exist, such as resampling with replacement or bootstrapping. This is often used in financial applications where we have a long time series of returns.

*   **Parametric Sampling:** This method involves estimating the parameters of a known probability distribution from historical data (e.g., estimating the mean and standard deviation of a normal distribution) or using expert knowledge. Then, instead of simply using the historical data points directly, we sample from the fitted distribution. This combines the benefits of historical data with the ability to generate scenarios beyond the observed range and potentially smooth out noise in the historical data.  It also provides a more concise representation of the uncertainty than simply using all historical data.

How to use it: Once the scenarios are generated using one of these methods (or a combination thereof), they are incorporated into the stochastic programming model. The model then optimizes a decision variable subject to constraints that hold (or are likely to hold) for each scenario. The objective function usually includes a risk measure or expected value calculation across all scenarios.

## 2) Application scenario

**Supply Chain Network Design Under Demand Uncertainty:**

Consider a company designing its supply chain network. They need to decide where to locate warehouses and how much inventory to hold at each location. The demand for their products is uncertain.

*   **Monte Carlo Sampling:**  Assume that demand in each region follows a normal distribution. Using historical sales data or market analysis, estimate the mean and standard deviation of demand for each region. Then, use Monte Carlo simulation to generate a set of demand scenarios for each region.

*   **Historical Sampling:**  Collect historical sales data for each region over the past few years.  Each year's demand data becomes a scenario. This approach doesn't assume any underlying distribution.

*   **Parametric Sampling:**  Fit a probability distribution (e.g., normal, gamma) to the historical sales data for each region. Then, sample from the fitted distribution to generate demand scenarios. This approach acknowledges historical trends but allows for extrapolation and capturing the overall shape of the data, rather than relying solely on past observations.

The stochastic programming model would then decide the optimal warehouse locations and inventory levels, considering the costs of opening warehouses, transportation costs, inventory holding costs, and potential penalties for unmet demand across all generated scenarios.
## 3) Python method (if possible)
```python
import numpy as np
import pandas as pd

# Example using numpy for scenario generation
def monte_carlo_scenario_generation(num_scenarios, mean, std_dev):
    """Generates scenarios using Monte Carlo simulation.

    Args:
        num_scenarios: The number of scenarios to generate.
        mean: The mean of the normal distribution.
        std_dev: The standard deviation of the normal distribution.

    Returns:
        A numpy array of scenarios.
    """
    scenarios = np.random.normal(loc=mean, scale=std_dev, size=num_scenarios)
    return scenarios

def historical_scenario_generation(historical_data):
  """Generates scenarios from historical data by simply using the data directly.

  Args:
      historical_data: A list or numpy array containing historical data.

  Returns:
      A numpy array containing the historical data as scenarios.
  """
  return np.array(historical_data)


def parametric_scenario_generation(num_scenarios, historical_data):
  """Generates scenarios by fitting a normal distribution to historical data and sampling from it.

  Args:
      num_scenarios: The number of scenarios to generate.
      historical_data: A list or numpy array containing historical data.

  Returns:
      A numpy array of scenarios.
  """
  mean = np.mean(historical_data)
  std_dev = np.std(historical_data)
  scenarios = np.random.normal(loc=mean, scale=std_dev, size=num_scenarios)
  return scenarios

# Example Usage
num_scenarios = 100

# Monte Carlo Example
mean_demand = 100
std_dev_demand = 20
mc_scenarios = monte_carlo_scenario_generation(num_scenarios, mean_demand, std_dev_demand)
print("Monte Carlo Scenarios (first 5):", mc_scenarios[:5])

# Historical Example
historical_demand = [80, 90, 110, 120, 100, 95, 105]
historical_scenarios = historical_scenario_generation(historical_demand)
print("Historical Scenarios:", historical_scenarios)

# Parametric Example
parametric_scenarios = parametric_scenario_generation(num_scenarios, historical_demand)
print("Parametric Scenarios (first 5):", parametric_scenarios[:5])

# Example using Pandas for Historical Scenario generation (easier to work with time series data)
# Simulate some daily demand data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
demand_values = np.random.randint(50, 150, size=365) # Random demand between 50 and 150
demand_data = pd.DataFrame({'Date': dates, 'Demand': demand_values})
demand_data = demand_data.set_index('Date')

# Now we can use historical sampling directly from the pandas DataFrame
historical_scenarios_pandas = demand_data['Demand'].values # numpy array
print("Historical Pandas Scenarios (first 5):", historical_scenarios_pandas[:5])
```

## 4) Follow-up question

How can scenario reduction techniques be used to reduce the computational burden associated with large scenario sets generated by Monte Carlo simulation, while still maintaining a representative set of scenarios for the stochastic programming model? Can you provide examples of common scenario reduction methods?
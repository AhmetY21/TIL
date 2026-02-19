---
title: "Application Pattern: Stochastic Inventory (newsvendor, multi-period)"
date: "2026-02-19"
week: 8
lesson: 6
slug: "application-pattern-stochastic-inventory-newsvendor-multi-period"
---

# Topic: Application Pattern: Stochastic Inventory (newsvendor, multi-period)

## 1) Formal definition (what is it, and how can we use it?)

Stochastic inventory models deal with managing inventory levels when future demand is uncertain. This section focuses on two key application patterns: the newsvendor problem (a single-period model) and multi-period inventory management.

*   **Newsvendor Problem:** This is a classic single-period inventory problem where a decision maker must decide how much to stock of a perishable or seasonal product before the selling season begins.  The key challenge is balancing the risk of overstocking (leftover inventory is sold at a loss or discarded) against the risk of understocking (lost sales and potential loss of customer goodwill). The objective is to maximize expected profit.  We can use the newsvendor model to determine the optimal order quantity. The optimal order quantity balances the cost of overstocking ($c_o$) and the cost of understocking ($c_u$). Specifically, the optimal fractile, *F(Q)*, representing the probability that demand is less than or equal to the optimal order quantity *Q*, is calculated as:

    *F(Q) = c_u / (c_u + c_o)*

    Where:
    *   *Q* is the optimal order quantity.
    *   *F(Q)* is the cumulative distribution function (CDF) of demand evaluated at *Q*.
    *   *c_u* is the cost of understocking (e.g., lost profit, cost of expedited orders).
    *   *c_o* is the cost of overstocking (e.g., salvage value, storage costs).

    If we know the distribution of demand and *c_u* and *c_o*, we can use this formula to find the corresponding *Q* that satisfies the equation.
*   **Multi-Period Inventory Management:** These models extend the newsvendor concept to multiple time periods. Decisions about how much to order in each period must consider current inventory levels, expected future demand, ordering costs (fixed and variable), holding costs, shortage costs, and lead times. These models are often formulated as stochastic dynamic programs or using simulation optimization techniques. Different policies exist, such as (s, S) policies (reorder when inventory drops below 's' to bring it up to 'S') or periodic review policies. The goal is to minimize total expected costs over the planning horizon.  We can use multi-period models to determine optimal ordering policies to balance competing cost factors when demand is uncertain.

## 2) Application scenario

*   **Newsvendor Problem:**  A retailer needs to decide how many Christmas trees to order in November. Demand for Christmas trees is uncertain and depends on factors such as weather, economic conditions, and local events.  If the retailer orders too many trees, they will have to be sold at a heavily discounted price or discarded.  If the retailer orders too few trees, they will miss out on potential sales and potentially lose customers to competitors.  *c_u* represents the profit lost on each missed sale, and *c_o* represents the difference between the purchase cost and the salvage value of unsold trees.

*   **Multi-Period Inventory Management:** A pharmaceutical company needs to manage its inventory of a particular drug. Demand for the drug fluctuates due to seasonal illnesses and marketing campaigns. The company faces fixed ordering costs each time it places an order, as well as holding costs for storing the drug in its warehouse and shortage costs if demand exceeds supply. The company must decide how much to order and when to order to minimize its total costs over the next year. They need to consider lead times (time between placing the order and receiving it) to ensure enough stock to fulfill orders.

## 3) Python method (if possible)

```python
import numpy as np
from scipy.stats import norm

def newsvendor_optimal_quantity(mean_demand, std_demand, cost_understock, cost_overstock):
  """
  Calculates the optimal order quantity for the newsvendor problem.

  Args:
    mean_demand: The mean of the demand distribution.
    std_demand: The standard deviation of the demand distribution.
    cost_understock: The cost of understocking (lost profit per unit).
    cost_overstock: The cost of overstocking (loss per unit).

  Returns:
    The optimal order quantity.
  """
  critical_ratio = cost_understock / (cost_understock + cost_overstock)
  optimal_quantity = norm.ppf(critical_ratio, loc=mean_demand, scale=std_demand)
  return optimal_quantity

# Example usage:
mean_demand = 100
std_demand = 20
cost_understock = 20  # Profit lost per unit if demand exceeds supply
cost_overstock = 5   # Loss per unit if supply exceeds demand

optimal_order_quantity = newsvendor_optimal_quantity(mean_demand, std_demand, cost_understock, cost_overstock)
print(f"Optimal order quantity: {optimal_order_quantity:.2f}")


# Example of a very basic multi-period inventory simulation.  This is a highly simplified example.
import random

def multi_period_inventory_simulation(demand_distribution, initial_inventory, order_quantity, holding_cost, shortage_cost, num_periods):
  """
  Simulates a multi-period inventory system.

  Args:
    demand_distribution: A function that returns a random demand value.
    initial_inventory: The starting inventory level.
    order_quantity: The fixed order quantity placed at the start of each period.
    holding_cost: The cost per unit of inventory held at the end of each period.
    shortage_cost: The cost per unit of demand not met in each period.
    num_periods: The number of periods to simulate.

  Returns:
    A dictionary containing the inventory levels, demand, and costs for each period.
  """
  inventory = initial_inventory
  results = {'inventory': [], 'demand': [], 'holding_cost': [], 'shortage_cost': []}

  for period in range(num_periods):
    # Place order at the beginning of the period (assuming no lead time for simplicity)
    inventory += order_quantity

    # Generate demand for the period
    demand = demand_distribution()
    results['demand'].append(demand)

    # Satisfy demand from inventory
    if inventory >= demand:
      inventory -= demand
      shortage = 0
    else:
      shortage = demand - inventory
      inventory = 0

    # Calculate costs
    holding_cost_period = holding_cost * inventory
    shortage_cost_period = shortage_cost * shortage

    # Store results
    results['inventory'].append(inventory)
    results['holding_cost'].append(holding_cost_period)
    results['shortage_cost'].append(shortage_cost_period)

  return results

# Example usage:
def demand_function():
  return random.randint(50, 150)  # Uniform demand between 50 and 150

initial_inventory = 50
order_quantity = 100
holding_cost = 1
shortage_cost = 10
num_periods = 10

results = multi_period_inventory_simulation(demand_function, initial_inventory, order_quantity, holding_cost, shortage_cost, num_periods)

print("Simulation Results:")
for i in range(num_periods):
  print(f"Period {i+1}: Inventory = {results['inventory'][i]}, Demand = {results['demand'][i]}, Holding Cost = {results['holding_cost'][i]}, Shortage Cost = {results['shortage_cost'][i]}")

total_holding_cost = sum(results['holding_cost'])
total_shortage_cost = sum(results['shortage_cost'])
print(f"\nTotal Holding Cost: {total_holding_cost}")
print(f"Total Shortage Cost: {total_shortage_cost}")
print(f"Total Cost: {total_holding_cost + total_shortage_cost}")

```

This Python code provides:

1.  **`newsvendor_optimal_quantity` function:**  Calculates the optimal order quantity using the critical ratio and the inverse cumulative distribution function (quantile function) of a normal distribution (using `scipy.stats.norm`).  It takes mean demand, standard deviation of demand, cost of understocking, and cost of overstocking as inputs.
2.  **`multi_period_inventory_simulation` function:** Simulates a basic multi-period inventory system. It includes placing an order at the beginning of each period, satisfying demand from inventory, and calculating holding and shortage costs.  It assumes a fixed order quantity and no lead time for simplicity. The demand distribution is provided as a function, allowing for flexibility. *Note:* This is a very simplistic simulation and lacks advanced features.
3.  **Example Usage:** Demonstrates how to use both functions with example parameters.

## 4) Follow-up question

How can simulation optimization techniques (e.g., using Optuna or similar libraries) be effectively applied to determine optimal (s, S) inventory policies in a multi-period stochastic inventory model when analytical solutions are not feasible, and what are the key considerations for setting up the simulation environment and the optimization objective?
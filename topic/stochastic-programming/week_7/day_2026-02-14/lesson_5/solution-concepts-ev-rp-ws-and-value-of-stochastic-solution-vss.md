---
title: "Solution Concepts: EV, RP, WS, and Value of Stochastic Solution (VSS)"
date: "2026-02-14"
week: 7
lesson: 5
slug: "solution-concepts-ev-rp-ws-and-value-of-stochastic-solution-vss"
---

# Topic: Solution Concepts: EV, RP, WS, and Value of Stochastic Solution (VSS)

## 1) Formal definition (what is it, and how can we use it?)

In Stochastic Programming, where optimization problems involve uncertainty represented by random variables, we often want to evaluate different solution approaches. Several solution concepts help us assess the impact of incorporating stochastic information into the decision-making process. These include:

*   **Expected Value (EV) Solution:** The EV solution involves replacing the random variables in the stochastic problem with their expected values and then solving the resulting deterministic problem. The resulting solution is denoted by x<sup>EV</sup>.

    *   **How to use it:** The EV solution serves as a baseline. It's computationally simple to obtain and represents what would happen if we ignored the uncertainty and simply assumed everything would be average. It is a "here and now" decision.

*   **Recourse Problem (RP) / Wait-and-See (WS) Solution:** The RP or WS approach involves solving the deterministic problem *after* the uncertainty has been revealed. For each possible realization of the random variables, we obtain an optimal solution, x(ω). Then, we average the optimal objective function values associated with each of these solutions to get the *Expected Result of Using the Recourse Problem (E[Q(x(ω), ω)])*.

    *   **How to use it:**  It gives the perfect hindsight performance – what could be achieved if we knew the future with certainty. We then compare the outcome of the EV Solution versus the outcome from averaging the solution across many iterations in RP. Often, in RP, one makes a "wait and see" decision, where we can adjust our actions depending on the realization of the random variable.

*   **Value of Stochastic Solution (VSS):** The VSS quantifies the benefit of using the stochastic programming solution (which considers the uncertainty) instead of the simpler expected value solution.  It is calculated as:

    VSS = E[Q(x<sup>EV</sup>, ω)] - E[Q(x(ω), ω)]

    where:
    *   E[Q(x<sup>EV</sup>, ω)] is the expected cost (or profit) when using the EV solution *under the real stochastic environment*.  We take the x<sup>EV</sup>, and then evaluate how it performs when input into the original stochastic problem. This is also known as the Expected Value of the Expected Value Solution, or *EEV*.
    *   E[Q(x(ω), ω)] is the expected cost (or profit) of solving the recourse problem (RP).

    *   **How to use it:** A positive VSS indicates that considering uncertainty leads to a better outcome (lower cost or higher profit on average) than simply using the expected value solution.  A high VSS justifies the effort to solve the more complex stochastic program.

In summary:

*   EV:  Solve the deterministic problem with expected values.  Easy, but ignores uncertainty.
*   RP/WS:  Solve many deterministic problems for each scenario and average the result. Perfect hindsight, but not practically realizable.
*   VSS:  Quantifies the gain from using a true stochastic solution (by comparing the performance of the EV solution under the stochastic reality with the performance of the RP solution).

## 2) Application scenario

Consider a supply chain problem where a company needs to decide how much of a product to order *now* before knowing the actual demand. The demand is uncertain and can be represented by a random variable.

*   **EV Solution:**  The company calculates the average demand and orders the quantity that would be optimal for that average demand. This is simple but might lead to overstocking if the actual demand is low or stockouts if the actual demand is high.
*   **Recourse Problem (RP):**  For each possible demand scenario, the company decides on the optimal quantity to order *after* observing the demand. This means if demand is high, they can quickly order more; if demand is low, they can reduce the excess stock. This is perfect hindsight.
*   **VSS:** The VSS would tell the company how much money they are saving (or losing) by using a more sophisticated stochastic programming model (which would consider multiple demand scenarios and associated probabilities) to determine the order quantity versus just ordering based on the average demand. A significant VSS would justify the use of the stochastic programming approach. For instance, a high VSS would indicate that a better solution to the problem would be a model that can account for multiple different demand scenarios.

## 3) Python method (if possible)

This example illustrates the concepts using a simple inventory management problem. We'll use `numpy` for calculations and `scipy.stats` for a normal distribution representing demand.

```python
import numpy as np
from scipy.stats import norm

# Define problem parameters
cost_per_unit = 1
selling_price_per_unit = 3
salvage_value_per_unit = 0.5
mean_demand = 100
std_dev_demand = 20
num_scenarios = 1000

# 1. Expected Value (EV) Solution
ev_demand = mean_demand
ev_order_quantity = ev_demand  # Simple assumption: order the expected demand
print(f"EV Order Quantity: {ev_order_quantity}")

def calculate_profit(order_quantity, demand):
    sales = min(order_quantity, demand)
    unsold = max(0, order_quantity - demand)
    profit = (sales * selling_price_per_unit) - (order_quantity * cost_per_unit) + (unsold * salvage_value_per_unit)
    return profit

# 2. Evaluate EV Solution under stochastic environment (EEV - Expected Value of the Expected Value Solution)
scenarios = norm.rvs(loc=mean_demand, scale=std_dev_demand, size=num_scenarios)
ev_profits = [calculate_profit(ev_order_quantity, demand) for demand in scenarios]
eev = np.mean(ev_profits)
print(f"EEV (Expected profit of EV solution under stochasticity): {eev}")


# 3. Recourse Problem (RP/WS) - Perfect Information
# For each scenario, find the best order quantity *knowing* the demand.  In this simple case, ordering exactly the demand maximizes profit.

rp_profits = [calculate_profit(demand, demand) for demand in scenarios]
rp_expected_profit = np.mean(rp_profits) #This is akin to finding out what the optimal profit is, for each scenario, and then finding the mean across all scenarios.
print(f"RP Expected Profit (Perfect Information): {rp_expected_profit}")

# 4. Value of Stochastic Solution (VSS)
vss = eev - rp_expected_profit
print(f"VSS: {vss}")


#Note: In this example with simple optimal RP strategy, the VSS will always be negative
```

**Explanation:**

*   The code simulates a simple inventory management problem.
*   It calculates the EV solution (ordering the expected demand).
*   It calculates the expected profit of using the EV solution under stochastic demand (EEV).
*   It then simulates the recourse problem (RP) by finding the optimal solution for each demand scenario (ordering exactly the demand).
*   Finally, it calculates the VSS.  The negative VSS indicates the EEV (or Expected Value of the EV solution) is a more profitable strategy than the recourse problem.

**Important Considerations:**

*   The optimality of ordering exactly the demand only holds for this simplified setting. In more complex settings with costs like backordering, this strategy is not necessarily optimal.
*   In the RP stage, we assumed that the optimum strategy involves ordering exactly the demand in each scenario. In general, one would need to solve another optimization problem in this stage as well.

## 4) Follow-up question

Suppose the code assumed it could order the exact demand for the recourse problem, but it could not. What if the recourse problem had a minimum order quantity of 10, no matter the demand?

How would this change impact the overall Value of Stochastic Solution?

```python
import numpy as np
from scipy.stats import norm

# Define problem parameters
cost_per_unit = 1
selling_price_per_unit = 3
salvage_value_per_unit = 0.5
mean_demand = 100
std_dev_demand = 20
num_scenarios = 1000

# 1. Expected Value (EV) Solution
ev_demand = mean_demand
ev_order_quantity = ev_demand  # Simple assumption: order the expected demand
print(f"EV Order Quantity: {ev_order_quantity}")

def calculate_profit(order_quantity, demand):
    sales = min(order_quantity, demand)
    unsold = max(0, order_quantity - demand)
    profit = (sales * selling_price_per_unit) - (order_quantity * cost_per_unit) + (unsold * salvage_value_per_unit)
    return profit

# 2. Evaluate EV Solution under stochastic environment (EEV - Expected Value of the Expected Value Solution)
scenarios = norm.rvs(loc=mean_demand, scale=std_dev_demand, size=num_scenarios)
ev_profits = [calculate_profit(ev_order_quantity, demand) for demand in scenarios]
eev = np.mean(ev_profits)
print(f"EEV (Expected profit of EV solution under stochasticity): {eev}")


# 3. Recourse Problem (RP/WS) - Perfect Information, but with a minimum order quantity
# For each scenario, find the best order quantity *knowing* the demand.  In this simple case, ordering the *max* of demand or 10 maximizes profit, accounting for the minimum order quantity.
min_order_quantity = 10

rp_profits = [calculate_profit(max(demand, min_order_quantity), demand) for demand in scenarios]
rp_expected_profit = np.mean(rp_profits)
print(f"RP Expected Profit (Perfect Information, min order quantity): {rp_expected_profit}")

# 4. Value of Stochastic Solution (VSS)
vss = eev - rp_expected_profit
print(f"VSS: {vss}")
```

**Impact:** The minimum order quantity constraint on the RP *decreases* the RP expected profit, because the optimal solution with perfect information cannot be achieved.

As a result, with the decrease in RP, the VSS will *increase*. This is because VSS = EEV - RP, therefore as RP decreases, VSS increases.

A higher VSS means that in this situation, that the difference between the simple EV method and the more nuanced stochastic method is greater. Therefore, with more constraints on RP, the better it is to actually employ the stochastic method, as its performance is relatively more advantageous than the EV method.
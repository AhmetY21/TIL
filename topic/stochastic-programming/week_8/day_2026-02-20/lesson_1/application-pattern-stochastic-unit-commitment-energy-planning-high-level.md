---
title: "Application Pattern: Stochastic Unit Commitment / Energy Planning (high level)"
date: "2026-02-20"
week: 8
lesson: 1
slug: "application-pattern-stochastic-unit-commitment-energy-planning-high-level"
---

# Topic: Application Pattern: Stochastic Unit Commitment / Energy Planning (high level)

## 1) Formal definition (what is it, and how can we use it?)

Stochastic Unit Commitment (SUC) and Stochastic Energy Planning are extensions of traditional unit commitment and energy planning problems that explicitly incorporate uncertainty.

*   **Traditional Unit Commitment (UC):** Aims to determine the optimal on/off schedule for generating units (e.g., power plants) to meet electricity demand at minimum cost, while respecting various operational constraints like ramp rates, minimum up/down times, and capacity limits. It typically assumes deterministic (fixed and known) demand and fuel prices.

*   **Traditional Energy Planning:**  Extends unit commitment to a longer time horizon (e.g., years, decades) and considers investment decisions in new generation capacity (e.g., building new power plants, renewable energy installations) alongside operational decisions. Similar to UC, it often relies on deterministic forecasts.

*   **Stochastic Unit Commitment (SUC):** Recognizes that electricity demand, renewable energy output (solar, wind), fuel prices, and even unit outages are uncertain. SUC uses probability distributions or scenarios to represent this uncertainty. The goal is to find a unit commitment schedule that minimizes the expected cost of operation *over a set of possible scenarios*, while still satisfying all constraints in each scenario.  It makes "here-and-now" commitment decisions (e.g., turning units on/off) that are robust against a wide range of potential future outcomes.

*   **Stochastic Energy Planning:** Builds upon stochastic UC by incorporating long-term investment decisions under uncertainty. It aims to optimize both investment in generation capacity and operational decisions, considering various scenarios for future demand growth, technology costs, environmental regulations, and fuel prices. It seeks the optimal mix of generation technologies and their operation over a long-term planning horizon, minimizing expected costs and risks.

**How we use it:**

SUC and Stochastic Energy Planning are used to:

*   **Improve System Reliability:**  By considering uncertainty, the solutions are more robust to unexpected events, reducing the risk of supply shortages or blackouts.
*   **Reduce Operating Costs:**  While seemingly counterintuitive, anticipating uncertainty can allow for more efficient dispatch and hedging strategies, potentially lowering overall operating costs compared to deterministic approaches.
*   **Facilitate Renewable Energy Integration:** Stochastic models are crucial for handling the variability and intermittency of renewable energy sources like solar and wind.
*   **Make Better Investment Decisions:** Stochastic Energy Planning provides a framework for evaluating different investment options under uncertainty, leading to more informed and resilient energy infrastructure development.
*   **Quantify Risk:** SUC and Stochastic Energy Planning provide insights into the potential risks associated with different operating and investment strategies, allowing decision-makers to make more informed choices.

## 2) Application scenario

Imagine a utility company planning its operations for the next 24 hours. The company has a mix of generation assets: coal-fired plants, natural gas turbines, and wind farms. The demand forecast for electricity is uncertain, and the wind farm output is highly variable and depends on unpredictable weather patterns.

**Deterministic UC Approach:** The utility would use a single, deterministic demand forecast and a single forecast of wind generation to create a unit commitment schedule. If the actual demand turns out to be higher than forecasted, or the wind output is lower, the system might face a shortage of electricity, requiring costly emergency purchases from the market or even load shedding (blackouts).

**Stochastic UC Approach:** The utility would create several "scenarios" representing different possibilities for demand and wind generation.  For example:

*   Scenario 1: High demand, low wind
*   Scenario 2: Medium demand, medium wind
*   Scenario 3: Low demand, high wind

The SUC model would then determine a unit commitment schedule that minimizes the *expected* operating cost across all these scenarios, while ensuring that demand is met in each scenario.  This might involve committing more flexible natural gas turbines to be ready to ramp up quickly if the wind output is low, even if it increases the expected cost slightly. The resulting schedule would be more robust to the uncertainties, reducing the risk of shortages and minimizing the overall long-term cost.

A similar application can be designed for Stochastic Energy Planning, but for a much longer time horizon (e.g., 20 years) and would include scenarios about fuel prices, policy, and new technologies.

## 3) Python method (if possible)

Using Python with a suitable optimization library like Pyomo, you can formulate and solve SUC and Stochastic Energy Planning problems. Here's a simplified example of a small SUC formulation.  It focuses on the core concepts and assumes a scenario-based approach.

```python
import pyomo.environ as pyo
import numpy as np

def create_suc_model(num_units, num_scenarios, demand_scenarios, unit_costs, unit_capacities):
    """
    Creates a Pyomo model for a simplified Stochastic Unit Commitment problem.

    Args:
        num_units (int): Number of generating units.
        num_scenarios (int): Number of scenarios.
        demand_scenarios (list): List of demand values for each scenario.
        unit_costs (list): List of cost per unit for each generating unit.
        unit_capacities (list): List of capacity for each generating unit.

    Returns:
        pyo.ConcreteModel: The Pyomo model.
    """

    model = pyo.ConcreteModel()

    # Sets
    model.Units = pyo.RangeSet(1, num_units)
    model.Scenarios = pyo.RangeSet(1, num_scenarios)

    # Parameters
    model.demand = pyo.Param(model.Scenarios, initialize=lambda model, s: demand_scenarios[s-1])
    model.cost = pyo.Param(model.Units, initialize=lambda model, u: unit_costs[u-1])
    model.capacity = pyo.Param(model.Units, initialize=lambda model, u: unit_capacities[u-1])
    model.scenario_probability = pyo.Param(model.Scenarios, initialize=1.0/num_scenarios) # Equal probability

    # Variables
    model.on = pyo.Var(model.Units, domain=pyo.Binary)  # Commitment decision (here-and-now)
    model.generation = pyo.Var(model.Units, model.Scenarios, domain=pyo.NonNegativeReals) # Generation level in each scenario

    # Objective Function
    def objective_rule(model):
        return sum(model.scenario_probability[s] * sum(model.cost[u] * model.generation[u, s] for u in model.Units) for s in model.Scenarios)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.MINIMIZE)

    # Constraints
    def demand_constraint_rule(model, s):
        return sum(model.generation[u, s] for u in model.Units) >= model.demand[s]
    model.demand_constraint = pyo.Constraint(model.Scenarios, rule=demand_constraint_rule)

    def capacity_constraint_rule(model, u, s):
        return model.generation[u, s] <= model.on[u] * model.capacity[u]
    model.capacity_constraint = pyo.Constraint(model.Units, model.Scenarios, rule=capacity_constraint_rule)

    return model

# Example usage
num_units = 2
num_scenarios = 3
demand_scenarios = [100, 120, 90]
unit_costs = [5, 7]  # Cost per unit generation
unit_capacities = [80, 60]

model = create_suc_model(num_units, num_scenarios, demand_scenarios, unit_costs, unit_capacities)

# Solve the model
solver = pyo.SolverFactory('glpk')  # You may need to install a solver like GLPK
solver.solve(model)

# Print results
print("Optimal Objective Value:", pyo.value(model.objective))
for u in model.Units:
    print(f"Unit {u} On/Off: {pyo.value(model.on[u])}")
    for s in model.Scenarios:
        print(f"  Scenario {s} Generation: {pyo.value(model.generation[u, s])}")
```

**Explanation:**

1.  **Model Creation:** The `create_suc_model` function sets up the Pyomo model.
2.  **Sets:** `Units` and `Scenarios` define the indices for the generating units and the scenarios, respectively.
3.  **Parameters:** `demand`, `cost`, `capacity`, and `scenario_probability` store the problem data.
4.  **Variables:** `on` (binary) indicates whether a unit is committed (on=1, off=0). This is a "here-and-now" decision, meaning it's made before knowing which scenario will occur. `generation` represents the generation level of each unit in each scenario.  This is a "wait-and-see" decision.
5.  **Objective Function:** Minimizes the expected cost of generation across all scenarios.
6.  **Constraints:**
    *   `demand_constraint`: Ensures that demand is met in each scenario.
    *   `capacity_constraint`: Limits the generation of each unit to its capacity, only if the unit is committed.
7.  **Solving:** The code uses the GLPK solver (you may need to install it using `apt-get install glpk` on linux or equivalent package managers on other OS) to solve the optimization problem. Other solvers (e.g., Gurobi, CPLEX) are often used for larger and more complex SUC problems.
8.  **Printing Results:** Displays the optimal objective value and the commitment and generation levels for each unit in each scenario.

**Limitations:**

*   **Simplification:** This is a highly simplified model. Real-world SUC models include many more constraints, such as ramp rates, minimum up/down times, reserve requirements, transmission constraints, and start-up costs.
*   **Solver:** GLPK might not be suitable for larger, more complex problems.
*   **Scenario Generation:** This example assumes scenarios are pre-defined. Real-world applications use more sophisticated techniques like Monte Carlo simulation or scenario reduction algorithms to generate representative scenarios.
*   **Two-Stage vs. Multi-Stage:** This example focuses on a two-stage approach (commitments are made now, generation levels are adjusted later). Multi-stage stochastic programming is often used for longer-term energy planning, allowing for sequential decision-making as uncertainty unfolds over time.

This simplified example provides a basic understanding of how stochastic unit commitment can be formulated and solved using Pyomo.  For real-world applications, you would need to expand the model to include more detailed constraints and use more sophisticated scenario generation and solution techniques.  Stochastic Energy Planning extends this framework to consider investment decisions over longer time horizons.

## 4) Follow-up question

How do different methods for scenario generation (e.g., Monte Carlo simulation, scenario reduction techniques) impact the computational complexity and accuracy of a stochastic unit commitment model? Which method is generally preferred in practice, and why?
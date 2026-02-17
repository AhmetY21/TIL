---
title: "Progressive Hedging Algorithm (PHA): scenario decomposition for NAC"
date: "2026-02-17"
week: 8
lesson: 3
slug: "progressive-hedging-algorithm-pha-scenario-decomposition-for-nac"
---

# Topic: Progressive Hedging Algorithm (PHA): scenario decomposition for NAC

## 1) Formal definition (what is it, and how can we use it?)

The Progressive Hedging Algorithm (PHA) is a scenario decomposition method used to solve stochastic programming problems, particularly those with Non-Anticipativity Constraints (NAC). Stochastic programming involves optimization problems where some parameters are uncertain and described by probability distributions. These distributions are often represented by a set of discrete scenarios. The NAC is a crucial component; it mandates that decisions made *before* the realization of uncertainty (first-stage decisions) must be the same across all scenarios.  In essence, it means we can't base our upfront decisions on information we don't yet have.

**What is it?**

PHA breaks down a large, complex stochastic program into a set of smaller, easier-to-solve subproblems, one for each scenario. It then iteratively coordinates the solutions of these subproblems to enforce the non-anticipativity constraint. This is done by penalizing deviations of the first-stage decisions across scenarios.  The algorithm aims to find a common first-stage decision that performs well across all scenarios by iteratively adjusting the solutions of individual scenario problems.

**How can we use it?**

Here's a breakdown of the iterative process:

1. **Initialization:**
   - Generate a set of scenarios, each with an associated probability.
   - Initialize the first-stage decision variables (denoted as `x`) for each scenario.  A common starting point is often used to ease the initial stages, but this is not required.
   - Set penalty parameters (also known as augmented Lagrangian multipliers, denoted as `ρ`) for the non-anticipativity constraints to initial values (often small positive values).

2. **Iteration (repeat until convergence):**
   - **Scenario Decomposition:** Solve each scenario problem independently, given the current first-stage decision variables (`x`) and penalty parameters (`ρ`). Each scenario problem typically includes the scenario-specific objective function, second-stage decisions, and constraints, *plus* a penalty term related to the difference between the scenario's first-stage decision and the average first-stage decision across all scenarios.
   - **Averaging:** Compute the average of the first-stage decision variables across all scenarios. This provides an estimate of the "consensus" decision.  Let `x_s` denote the first-stage decision for scenario `s`, then the average is `x_bar = sum(p_s * x_s)` where `p_s` is the probability of scenario `s`.
   - **Updating First-Stage Decisions:** For each scenario, update the first-stage decision variables based on the scenario solution and the average decision. This update typically involves shifting the first-stage decision towards the average, weighted by the penalty parameter `ρ`.
   - **Updating Penalty Parameters (Augmented Lagrangian multipliers):** Update the penalty parameters `ρ`.  This usually involves increasing `ρ` when the non-anticipativity constraints are violated (i.e., the difference between the scenario's first-stage decision and the average decision is above a certain tolerance).

3. **Convergence:** Check for convergence. Convergence is usually determined by checking if the difference between the first-stage decisions across scenarios is small enough (i.e., the non-anticipativity constraints are approximately satisfied) and/or if the objective function value is not changing significantly between iterations.

PHA is particularly useful for problems with a large number of scenarios where solving the entire stochastic program at once is computationally intractable. The decomposition allows for parallelization, making it suitable for high-performance computing environments.

## 2) Application scenario

Consider a power generation company that needs to decide how much to invest in different types of energy sources (e.g., solar, wind, natural gas) to meet future electricity demand. The future electricity demand and fuel prices are uncertain and represented by a set of scenarios.

*   **First-Stage Decision (x):** Investment decisions (e.g., capacity of each energy source to build). These decisions must be made *now*, before we know the actual future demand and prices.
*   **Second-Stage Decisions (y):** Operational decisions (e.g., how much electricity to generate from each source in each period). These decisions are made *after* the scenario is revealed and can be scenario-dependent.
*   **Uncertainty:** Future electricity demand, fuel prices, and renewable energy availability.
*   **Objective:** Minimize the total cost of investment and operation over a planning horizon, subject to meeting demand in each scenario.
*   **Non-Anticipativity Constraint:** The investment decisions (first-stage variables) must be the same across all scenarios.  We can't build a different power plant portfolio based on what we *think* might happen in each scenario.

Applying PHA to this problem involves:

1.  Creating a set of scenarios representing different possible realizations of future demand and fuel prices.
2.  Solving a separate optimization problem for each scenario, where the objective is to minimize the cost of meeting demand in that scenario, subject to the investment decisions (first-stage variables) being equal to some initial guess.
3.  Averaging the investment decisions across all scenarios.
4.  Updating the investment decisions for each scenario based on the average decision and the penalty parameters.
5.  Updating the penalty parameters.
6.  Repeating steps 2-5 until the investment decisions converge across scenarios.

## 3) Python method (if possible)

Here's a simplified Python example using the `Pyomo` library for optimization. Note that this is a high-level illustration and would require more detailed modeling for a real-world application.  This example assumes that the scenario generation has already occurred and that we are starting from a set of scenarios ready for PHA.  Key aspects like solver choice, parameter tuning, and more complex convergence criteria are omitted for clarity.

```python
import pyomo.environ as pyo
import numpy as np

def progressive_hedging(scenarios, rho_initial=1.0, rho_update_factor=1.1, tolerance=1e-6, max_iterations=100):
    """
    Implements the Progressive Hedging Algorithm.

    Args:
        scenarios (list): A list of Pyomo model instances, one for each scenario.
                         Each model should have first-stage variables named 'x' and
                         an objective function.
        rho_initial (float): Initial value for the penalty parameter (rho).
        rho_update_factor (float): Factor by which to increase rho if NAC is violated.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: List of updated Pyomo model instances, one for each scenario, after PHA.
    """

    num_scenarios = len(scenarios)
    probabilities = [1/num_scenarios] * num_scenarios  # Assuming equal probabilities
    rho = rho_initial
    x_avg = None #initialize the average, it will be calculated at the first iteration

    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}/{max_iterations}")

        # 1. Solve scenario problems
        x_solutions = []
        for s in range(num_scenarios):
            # Add augmented Lagrangian term to the objective function:
            # Minimize OriginalObjective + rho/2 * sum((x[i] - x_avg[i])**2)
            # + sum(dual_vars[i] * (x[i] - x_avg[i])) #Dual Variable
            model = scenarios[s]
            if iteration > 0:
                penalty_term = 0
                for var in model.x:  # Assuming 'x' is a Set for the first stage vars
                    penalty_term += rho/2 * (model.x[var] - x_avg[var])**2

                model.objective.expr = model.objective.expr + penalty_term

            # Solve the problem
            solver = pyo.SolverFactory('glpk') #or any other solver
            solver.solve(model)
            x_solutions.append({v: pyo.value(model.x[v]) for v in model.x})

        # 2. Calculate average first-stage decision
        x_avg = {}
        for var in scenarios[0].x: #assumes all models share the same first-stage variables
            x_avg[var] = sum(probabilities[s] * x_solutions[s][var] for s in range(num_scenarios))


        # 3. Check for convergence
        max_deviation = 0
        for s in range(num_scenarios):
            for var in scenarios[0].x:
                deviation = abs(x_solutions[s][var] - x_avg[var])
                max_deviation = max(max_deviation, deviation)


        print(f"Max deviation: {max_deviation}")
        if max_deviation < tolerance:
            print("Convergence achieved!")
            break

        # 4. Update rho (if needed - omitted here for simplicity, see follow-up)
        if max_deviation > tolerance:
            rho *= rho_update_factor #Increase rho
            print(f"Updating rho to {rho}")
        else:
            print("no penalty update, small max deviation")


    return scenarios

# Example Usage (replace with your actual models):
# First, define a sample Stochastic Problem with two stages
def create_scenario_model(scenario_data): # a scenario could be a single dictionary, as you define it
  model = pyo.ConcreteModel()

  # Index set for first-stage variables (example: product types)
  model.PRODUCTS = pyo.Set(initialize=['A', 'B'])

  # First-stage variables (e.g., production quantities)
  model.x = pyo.Var(model.PRODUCTS, within=pyo.NonNegativeReals, name='x')

  # Second-stage variables (e.g., sales quantities)
  model.y = pyo.Var(model.PRODUCTS, within=pyo.NonNegativeReals)

  # Parameters (scenario-dependent)
  model.demand = pyo.Param(model.PRODUCTS, initialize=scenario_data['demand'])
  model.price = pyo.Param(model.PRODUCTS, initialize=scenario_data['price'])
  model.production_cost = pyo.Param(model.PRODUCTS, initialize=scenario_data['production_cost'])

  # Objective function
  def objective_rule(model):
      production_costs = sum(model.production_cost[p] * model.x[p] for p in model.PRODUCTS)
      revenues = sum(model.price[p] * model.y[p] for p in model.PRODUCTS)
      return production_costs - revenues

  model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

  # Constraints
  def sales_limit_rule(model, p):
      return model.y[p] <= model.demand[p]
  model.sales_limit = pyo.Constraint(model.PRODUCTS, rule=sales_limit_rule)

  def production_limit_rule(model, p):
    return model.y[p] <= model.x[p]
  model.production_limit = pyo.Constraint(model.PRODUCTS, rule = production_limit_rule)

  return model

# Create a dummy set of scenarios
scenario_data1 = {'demand': {'A': 10, 'B': 15}, 'price': {'A': 5, 'B': 7}, 'production_cost':{'A':1, 'B':2}}
scenario_data2 = {'demand': {'A': 12, 'B': 13}, 'price': {'A': 6, 'B': 6}, 'production_cost':{'A':1.5, 'B':2.5}}
scenarios = [create_scenario_model(scenario_data1), create_scenario_model(scenario_data2)]

# Run PHA
updated_scenarios = progressive_hedging(scenarios)

# Print results for the first scenario (for demonstration)
print("\nResults for Scenario 1:")
for var in updated_scenarios[0].x:
    print(f"x[{var}]: {pyo.value(updated_scenarios[0].x[var])}")
```

**Important notes:**

*   This code provides a conceptual illustration. A robust implementation requires careful tuning of parameters like `rho_initial`, `rho_update_factor`, and `tolerance`.
*   The specific structure of your `scenarios` list will depend on how your stochastic program is modeled using Pyomo.  The code assumes the first-stage variables are accessed via `model.x`.
*   This example uses `glpk` solver which is not the most efficient, consider other solvers.
*   Consider a more sophisticated method for updating `rho` like using a dual variable.
*  Consider implementing a non-increasing penalty for augmented lagrangian (e.g., see https://link.springer.com/article/10.1007/s10589-022-00404-x)

## 4) Follow-up question

How can I implement a more sophisticated rho update scheme that utilizes the dual variables from solving each scenario's subproblem? This would allow for a more dynamic adjustment of the penalty parameter based on the actual violation of the non-anticipativity constraints, potentially leading to faster convergence. Show how the Augmented Lagrangian Dual variable would be defined in `pyomo` within the above `create_scenario_model` definition. Then, show how to update rho based on a dual variable within the `progressive_hedging` function.

To answer the question, the augmented lagrangian would be set up using the first stage variables. In this example, they are indexed by `model.PRODUCTS` as shown above. Add the following to the function:
1) Augmented Lagrangian variable in the `create_scenario_model`
2) Include the `dual_vars` as part of solving the objective
3) Update the rho parameter in the main loop using the dual variables.

```python
import pyomo.environ as pyo
import numpy as np

def progressive_hedging(scenarios, rho_initial=1.0, rho_update_factor=1.1, tolerance=1e-6, max_iterations=100, dual_update_step = 0.1):
    """
    Implements the Progressive Hedging Algorithm with dual variable updates.

    Args:
        scenarios (list): A list of Pyomo model instances, one for each scenario.
                         Each model should have first-stage variables named 'x' and
                         an objective function, and dual variables 'dual'.
        rho_initial (float): Initial value for the penalty parameter (rho).
        rho_update_factor (float): Factor by which to increase rho if NAC is violated.
        tolerance (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.
        dual_update_step (float): Step size for updating dual variables.

    Returns:
        list: List of updated Pyomo model instances, one for each scenario, after PHA.
    """

    num_scenarios = len(scenarios)
    probabilities = [1/num_scenarios] * num_scenarios  # Assuming equal probabilities
    rho = rho_initial
    x_avg = None #initialize the average, it will be calculated at the first iteration

    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}/{max_iterations}")

        # 1. Solve scenario problems
        x_solutions = []
        for s in range(num_scenarios):
            # Add augmented Lagrangian term to the objective function:
            # Minimize OriginalObjective + rho/2 * sum((x[i] - x_avg[i])**2)
            # + sum(dual_vars[i] * (x[i] - x_avg[i])) #Dual Variable
            model = scenarios[s]
            if iteration > 0:
                penalty_term = 0
                dual_term = 0
                for var in model.x:  # Assuming 'x' is a Set for the first stage vars
                    penalty_term += rho/2 * (model.x[var] - x_avg[var])**2
                    dual_term += model.dual[var] * (model.x[var] - x_avg[var])

                model.objective.expr = model.objective.expr + penalty_term + dual_term

            # Solve the problem
            solver = pyo.SolverFactory('glpk') #or any other solver
            solver.solve(model)
            x_solutions.append({v: pyo.value(model.x[v]) for v in model.x})

        # 2. Calculate average first-stage decision
        x_avg = {}
        for var in scenarios[0].x: #assumes all models share the same first-stage variables
            x_avg[var] = sum(probabilities[s] * x_solutions[s][var] for s in range(num_scenarios))


        # 3. Check for convergence
        max_deviation = 0
        for s in range(num_scenarios):
            for var in scenarios[0].x:
                deviation = abs(x_solutions[s][var] - x_avg[var])
                max_deviation = max(max_deviation, deviation)


        print(f"Max deviation: {max_deviation}")
        if max_deviation < tolerance:
            print("Convergence achieved!")
            break

        # 4. Update dual variables
        for s in range(num_scenarios):
          model = scenarios[s]
          for var in model.x:
              model.dual[var] = model.dual[var] + rho * (pyo.value(model.x[var]) - x_avg[var])


        # 5. Update rho (if needed - omitted here for simplicity, see follow-up)
        if max_deviation > tolerance:
            rho *= rho_update_factor #Increase rho
            print(f"Updating rho to {rho}")
        else:
            print("no penalty update, small max deviation")


    return scenarios

# Example Usage (replace with your actual models):
# First, define a sample Stochastic Problem with two stages
def create_scenario_model(scenario_data): # a scenario could be a single dictionary, as you define it
  model = pyo.ConcreteModel()

  # Index set for first-stage variables (example: product types)
  model.PRODUCTS = pyo.Set(initialize=['A', 'B'])

  # First-stage variables (e.g., production quantities)
  model.x = pyo.Var(model.PRODUCTS, within=pyo.NonNegativeReals, name='x')

  # Second-stage variables (e.g., sales quantities)
  model.y = pyo.Var(model.PRODUCTS, within=pyo.NonNegativeReals)

  # Augmented Lagrangian dual variables
  model.dual = pyo.Var(model.PRODUCTS, initialize=0, name='dual') #initialize dual to 0

  # Parameters (scenario-dependent)
  model.demand = pyo.Param(model.PRODUCTS, initialize=scenario_data['demand'])
  model.price = pyo.Param(model.PRODUCTS, initialize=scenario_data['price'])
  model.production_cost = pyo.Param(model.PRODUCTS, initialize=scenario_data['production_cost'])

  # Objective function
  def objective_rule(model):
      production_costs = sum(model.production_cost[p] * model.x[p] for p in model.PRODUCTS)
      revenues = sum(model.price[p] * model.y[p] for p in model.PRODUCTS)
      return production_costs - revenues

  model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

  # Constraints
  def sales_limit_rule(model, p):
      return model.y[p] <= model.demand[p]
  model.sales_limit = pyo.Constraint(model.PRODUCTS, rule=sales_limit_rule)

  def production_limit_rule(model, p):
    return model.y[p] <= model.x[p]
  model.production_limit = pyo.Constraint(model.PRODUCTS, rule = production_limit_rule)

  return model

# Create a dummy set of scenarios
scenario_data1 = {'demand': {'A': 10, 'B': 15}, 'price': {'A': 5, 'B': 7}, 'production_cost':{'A':1, 'B':2}}
scenario_data2 = {'demand': {'A': 12, 'B': 13}, 'price': {'A': 6, 'B': 6}, 'production_cost':{'A':1.5, 'B':2.5}}
scenarios = [create_scenario_model(scenario_data1), create_scenario_model(scenario_data2)]

# Run PHA
updated_scenarios = progressive_hedging(scenarios)

# Print results for the first scenario (for demonstration)
print("\nResults for Scenario 1:")
for var in updated_scenarios[0].x:
    print(f"x[{var}]: {pyo.value(updated_scenarios[0].x[var])}")

```
**Explanation of the changes:**

1.  **Augmented Lagrangian Dual Variables:**

    *   In `create_scenario_model`, a new `pyo.Var` called `dual` is added, indexed by `model.PRODUCTS`. This represents the dual variables associated with the non-anticipativity constraints for each first-stage variable. The initial value is set to 0.

2.  **Including the Dual Term in the Objective:**

    *   In `progressive_hedging`, within the main loop, the `dual_term` is added to the objective function of each scenario. The `dual_term` is computed as the sum of `model.dual[var] * (model.x[var] - x_avg[var])` for each first-stage variable `var`.
    *   The complete objective function now includes the original objective, the quadratic penalty term (augmented Lagrangian penalty), and the dual term.

3.  **Updating Dual Variables:**

    *   After solving each scenario and checking for convergence, the dual variables are updated using the following formula: `model.dual[var] = model.dual[var] + rho * (pyo.value(model.x[var]) - x_avg[var])`. This is a standard dual update rule, where the dual variable is adjusted based on the violation of the non-anticipativity constraint (the difference between the scenario's first-stage variable and the average).

This implementation should enable a more adaptive penalty adjustment, potentially leading to faster convergence compared to a fixed or simply increasing `rho`. However, parameter tuning remains crucial for optimal performance. Specifically, the `dual_update_step` should be tested. Also, since this requires computing the dual variable, the appropriate pyomo solver that computes it must be selected.
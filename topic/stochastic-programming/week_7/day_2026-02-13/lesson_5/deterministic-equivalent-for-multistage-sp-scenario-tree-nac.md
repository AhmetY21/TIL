---
title: "Deterministic Equivalent for multistage SP (scenario tree + NAC)"
date: "2026-02-13"
week: 7
lesson: 5
slug: "deterministic-equivalent-for-multistage-sp-scenario-tree-nac"
---

# Topic: Deterministic Equivalent for multistage SP (scenario tree + NAC)

## 1) Formal definition (what is it, and how can we use it?)

The deterministic equivalent of a multistage stochastic program with a scenario tree and non-anticipativity constraints (NACs) is a large-scale, deterministic optimization problem that, when solved, yields the optimal decisions for each stage and scenario of the original stochastic program. It effectively "unravels" the uncertainty by explicitly modeling all possible scenario outcomes and imposing constraints that ensure decisions made before the realization of uncertainty are consistent across all scenarios that share the same history up to that point.

**Here's a breakdown:**

*   **Scenario Tree:** A scenario tree represents the evolution of uncertainty over time.  Nodes in the tree represent states of the world at different time stages.  Edges represent the realization of uncertain events that lead from one state to the next. Each path from the root node (time 0) to a leaf node represents a unique scenario.

*   **Multistage Stochastic Program:** A stochastic program where decisions are made sequentially over time, with uncertainty revealed between decision stages.  Decisions at each stage must account for the possible future outcomes.

*   **Non-Anticipativity Constraints (NACs):** These are crucial.  They enforce the requirement that decisions made in early stages *must be the same* across all scenarios that share the same history up to that stage. In other words, you can't use information about the future (which you don't have yet) when making decisions in the present. NACs are typically formulated as equality constraints, forcing decision variables corresponding to the same history to have the same value.  Formally: `x_t(s1) = x_t(s2)` if scenarios `s1` and `s2` have the same history up to time `t`.  `x_t(s)` is the decision variable at time `t` in scenario `s`.

*   **Deterministic Equivalent:** The resulting deterministic optimization problem is formed by:
    1.  Defining decision variables for each stage and scenario.
    2.  Writing out the objective function, which is a weighted sum of the objective function values for each scenario (weighted by the scenario probability). This typically involves expected cost minimization or expected profit maximization.
    3.  Writing out the constraints of the original stochastic program for each scenario. These constraints might involve resource limitations, demand satisfaction, or physical laws.
    4.  Adding the non-anticipativity constraints.

**How can we use it?**

The deterministic equivalent allows us to solve a stochastic program using standard deterministic optimization solvers (e.g., linear programming solvers, mixed-integer programming solvers). By formulating the problem as a large deterministic optimization problem, we can leverage efficient algorithms and software to find the optimal solution. However, the size of the deterministic equivalent grows exponentially with the number of stages and scenarios, potentially leading to computational challenges for large-scale problems.

## 2) Application scenario

**Example: Supply Chain Planning under Demand Uncertainty**

Imagine a company needs to plan its production and distribution of goods over three time periods (months).  The demand for the product is uncertain and is represented by a scenario tree.

*   **Stage 1 (Month 1):** The company must decide how much to produce and where to ship it *before* knowing the actual demand.
*   **Uncertainty Revealed:** At the end of Month 1, the demand for the product in Months 2 and 3 is revealed, leading to different possible scenarios.
*   **Stage 2 (Month 2):** Based on the realized demand for Month 2 and the remaining demand forecast for Month 3, the company adjusts its production and distribution plan.
*   **Uncertainty Revealed:** At the end of Month 2, the demand for Month 3 is revealed.
*   **Stage 3 (Month 3):** The company makes final production and distribution decisions to meet the realized demand in Month 3.

**Scenario Tree:**  A simple scenario tree might have 3 stages and, say, 4 scenarios.
    *   Stage 0 (root): Initial state
    *   Stage 1: Decisions about production levels for the first month are made.
    *   Stage 2: Uncertainty is revealed. Two branches: High demand, Low demand.
    *   Stage 3: Uncertainty further revealed. High/High, High/Low, Low/High, Low/Low. Now we have 4 branches.

**Deterministic Equivalent Formulation:**

*   **Variables:**
    *   `x_t(s)`: Production quantity at stage `t` in scenario `s`.
    *   `y_t(s)`: Distribution quantity from plant to customer at stage `t` in scenario `s`.
    *   `I_t(s)`: Inventory level at stage `t` in scenario `s`.

*   **Objective Function:** Minimize the expected total cost, including production costs, inventory holding costs, and transportation costs, across all scenarios:  `Minimize SUM_s (probability(s) * (cost_production(x(s)) + cost_inventory(I(s)) + cost_transport(y(s))))`

*   **Constraints:**
    *   **Production Capacity Constraints:** `x_t(s) <= capacity_t` for all `t`, `s`.
    *   **Demand Satisfaction Constraints:** `SUM_plant y_t(s) = demand_t(s)` for all `t`, `s`.  `demand_t(s)` is the realized demand in scenario `s` at time `t`.
    *   **Inventory Balance Constraints:** `I_t(s) = I_{t-1}(s) + x_t(s) - SUM_customer y_t(s)` for all `t`, `s`.
    *   **Non-Negativity Constraints:** `x_t(s) >= 0`, `y_t(s) >= 0`, `I_t(s) >= 0` for all `t`, `s`.
    *   **Non-Anticipativity Constraints:** For example, `x_1(s1) = x_1(s2)` for all scenarios `s1` and `s2` because decisions at time 1 must be the same regardless of the future demand.  Specifically, all scenarios that branch from stage 0 must have the same first-stage decision. Similar NACs apply to the distribution variables `y` at the appropriate stages.

Solving this deterministic equivalent problem will provide the optimal production and distribution plan for each stage and scenario, while respecting the non-anticipativity constraints.

## 3) Python method (if possible)

```python
import pyomo.environ as pyo

def create_deterministic_equivalent(scenario_tree, model_callback):
    """
    Creates a deterministic equivalent of a multistage stochastic program.

    Args:
        scenario_tree (dict): A dictionary representing the scenario tree.
                              Keys are node names. Values are dictionaries containing:
                              - 'stage': The stage number (0, 1, 2, ...).
                              - 'probability': The probability of the scenario (for leaf nodes).
                              - 'children': A list of child node names (for non-leaf nodes).

        model_callback (function): A function that takes a scenario node and a Pyomo model
                                   as input and defines the variables, objective, and
                                   stage-specific constraints for that scenario.

    Returns:
        pyo.ConcreteModel: A Pyomo ConcreteModel representing the deterministic equivalent.
    """

    model = pyo.ConcreteModel()

    # Create a set of all scenario nodes
    model.SCENARIOS = pyo.Set(initialize=scenario_tree.keys())

    # Scenario probability parameter (only for leaf nodes)
    def scenario_probability_init(model, s):
        return scenario_tree[s].get('probability', 1.0) if 'probability' in scenario_tree[s] else 1.0
    model.scenario_probability = pyo.Param(model.SCENARIOS, initialize=scenario_probability_init, mutable=False)


    # Create scenario-specific variables and constraints
    for s in model.SCENARIOS:
        model_callback(scenario_tree[s], model, s) # Pass scenario name

    # Non-Anticipativity Constraints (NACs)
    def create_nacs(model, stage):
      scenarios_in_stage = [s for s in model.SCENARIOS if scenario_tree[s]['stage'] == stage]
      if not scenarios_in_stage:
        return pyo.Constraint.Skip
      
      #Assuming all variables in stage have the same prefix e.g x_1_high, x_1_low, x_2_high etc.
      #this example assumes variables begin with x
      
      # Find variables for comparison, assuming they are named x_stage_scenario
      first_scenario = scenarios_in_stage[0]
      variables_to_compare = [var_name for var_name in model.component_map(pyo.Var).keys()
                               if var_name.startswith(f'x_{stage}_{first_scenario}')]
      
      for var_name_prefix in variables_to_compare:
          # Get base name without scenario suffix
          #base_var_name = var_name_prefix[:-len(first_scenario) - 1] # Strip stage and scenario suffix
          base_var_name = var_name_prefix.split('_') # split name
          base_var_name = '_'.join(base_var_name[:-1]) #remove scenario identifier
          base_var_name = base_var_name.split('_') # split name
          base_var_name = '_'.join(base_var_name[:-1]) #remove stage identifier
          
          
          
          for i in range(1, len(scenarios_in_stage)):
              scenario1 = scenarios_in_stage[0]
              scenario2 = scenarios_in_stage[i]
              # Construct variable names for both scenarios
              var1_name = f"{base_var_name}_{stage}_{scenario1}"
              var2_name = f"{base_var_name}_{stage}_{scenario2}"

              # Check if the variables exist in the model
              if hasattr(model, var1_name) and hasattr(model, var2_name):
                  var1 = getattr(model, var1_name)
                  var2 = getattr(model, var2_name)

                  # Create non-anticipativity constraint
                  nac_name = f"NAC_{var1_name}_{var2_name}"
                  setattr(model, nac_name, pyo.Constraint(expr=var1 == var2))


    # Apply non-anticipativity constraints for all stages (except the last)
    max_stage = max(scenario_tree[s]['stage'] for s in scenario_tree)
    for stage in range(max_stage):
      create_nacs(model, stage+1)


    return model


# Example usage (replace with your specific model and data):
if __name__ == '__main__':

    # Define a simple scenario tree
    scenario_tree = {
        'root': {'stage': 0, 'children': ['high', 'low']},
        'high': {'stage': 1, 'probability': 0.6, 'children': []},
        'low': {'stage': 1, 'probability': 0.4, 'children': []},
    }

    def model_callback(scenario_node, model, scenario_name): #take scenario name as input
        stage = scenario_node['stage']
        # Define variables for each scenario
        # Scenario 0 and 1 in the stage.
        setattr(model, f"x_{stage}_{scenario_name}", pyo.Var(within=pyo.NonNegativeReals)) #create scenario specific variables, and set var names
        setattr(model, f"demand_{stage}_{scenario_name}", pyo.Param(initialize=10 if scenario_name == 'high' else 5))

        # Define a dummy objective (replace with your actual objective)
        if not hasattr(model, 'objective'): #avoid multiple calls
            model.objective = pyo.Objective(
                expr=sum(model.scenario_probability[s] * getattr(model, f"x_{scenario_tree[s]['stage']}_{s}") for s in model.SCENARIOS),
                sense=pyo.minimize
            )

        # Define dummy constraints
        def demand_constraint_rule(model):
            return getattr(model, f"x_{stage}_{scenario_name}") >= getattr(model, f"demand_{stage}_{scenario_name}")
        setattr(model, f"demand_constraint_{scenario_name}", pyo.Constraint(rule=demand_constraint_rule))

    # Create the deterministic equivalent
    deterministic_model = create_deterministic_equivalent(scenario_tree, model_callback)

    # Solve the model
    solver = pyo.SolverFactory('glpk') # or any other solver
    solver.solve(deterministic_model)

    # Print the solution
    for v in deterministic_model.component_objects(pyo.Var):
        print(f"{v.name}: {v.get_values()}")
```

**Explanation:**

1.  **`create_deterministic_equivalent(scenario_tree, model_callback)` function:**
    *   Takes the scenario tree (dictionary) and a `model_callback` function as input.
    *   Creates a Pyomo `ConcreteModel`.
    *   Defines a `SCENARIOS` set based on the keys in the `scenario_tree`.
    *   The `model_callback` function is called for *each* scenario node in the tree.  This function should add scenario-specific variables, constraints, and contribute to the overall objective function.
    *   Implements the non-anticipativity constraints, by looping through all the variables that have the same prefix. This code also allows you to easily change/alter the non-anticipativity contraints.
    *   Returns the deterministic equivalent Pyomo model.

2.  **`model_callback(scenario_node, model, scenario_name)` function:**
    *   This is the *crucial* part where you define your specific stochastic program's components *for each scenario*.
    *   It receives the scenario node's data, the Pyomo model, and the scenario name.
    *   Inside this function:
        *   Define your decision variables (e.g., production levels, inventory levels) for that specific scenario.  *Important*: Give them unique names that include the scenario name (e.g., `production_high`, `production_low`). I have named the variables as `x_stage_scenario`.
        *   Define any constraints that apply *specifically* to that scenario (e.g., demand satisfaction constraints, resource limitations).
        *   Update/append to the objective function.

3.  **Example Usage:**
    *   A simple scenario tree is defined.
    *   The `model_callback` function is implemented (replace the dummy objective and constraints with your actual model).
    *   `create_deterministic_equivalent()` is called to create the deterministic model.
    *   The deterministic model is solved using a solver (GLPK in this example).
    *   The solution is printed.

**Important Considerations:**

*   **Variable Naming:** The variable naming convention in the `model_callback` is very important.  It must be consistent and allow the non-anticipativity constraints to be correctly identified and applied. The above python code shows an example, where the naming convention is: `x_stage_scenario`
*   **Solver Choice:** The size of the deterministic equivalent can become very large. Choose a solver appropriate for the size and type (linear, integer, etc.) of your problem.
*   **Scaling:** For large-scale problems, decomposition techniques (e.g., Benders decomposition, progressive hedging) might be necessary to solve the stochastic program more efficiently than solving the full deterministic equivalent directly.
*   **Adaptation:** The `model_callback` and `create_deterministic_equivalent` functions will need to be adapted to your specific stochastic program's structure and data.  The example provided is a starting point.

## 4) Follow-up question

How can decomposition methods, like Benders Decomposition or Progressive Hedging, be integrated with this deterministic equivalent framework to solve larger stochastic programming problems that are computationally intractable to solve directly?  Specifically, how would the non-anticipativity constraints be handled within these decomposition algorithms?
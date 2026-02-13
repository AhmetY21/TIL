---
title: "Deterministic Equivalent for multistage SP (scenario tree + NAC)"
date: "2026-02-13"
week: 7
lesson: 6
slug: "deterministic-equivalent-for-multistage-sp-scenario-tree-nac"
---

# Topic: Deterministic Equivalent for multistage SP (scenario tree + NAC)

## 1) Formal definition (what is it, and how can we use it?)

The "Deterministic Equivalent" of a multistage Stochastic Program (SP) with a scenario tree and Non-Anticipativity Constraints (NACs) is a large-scale deterministic optimization problem that, when solved, yields the optimal decisions for all stages and scenarios of the original stochastic problem. It essentially transforms the stochastic problem into a deterministic one by explicitly representing all possible scenarios and their interdependencies.

**Components:**

*   **Scenario Tree:** A directed tree structure that represents the evolution of uncertainty over time. Each node represents a state or "scenario" at a particular stage. The root node represents the initial state, and branches emanating from a node represent possible realizations of uncertain parameters.  Each path from the root to a leaf node represents a complete scenario.

*   **Non-Anticipativity Constraints (NACs):**  These constraints enforce the principle that decisions made at a particular stage cannot depend on future information that is not yet available at that stage.  In other words, decisions made *before* a particular branch point in the scenario tree must be the same for all scenarios that are indistinguishable up to that point.  NACs typically equate decision variables across scenarios that share the same history.

*   **Deterministic Equivalent Formulation:**
    1.  **Variables:** Decision variables are indexed by stage and scenario (or node in the scenario tree).  So, we have variables like `x_{t,s}` representing the decision `x` at stage `t` in scenario `s`.
    2.  **Objective Function:** The objective function is a weighted sum of the objective values in each scenario, where the weights are the probabilities of each scenario occurring. This represents the expected value of the objective over all possible scenarios.  Often it minimizes costs or maximizes profits.
    3.  **Constraints:**
        *   **Stage-specific constraints:** These are the constraints that apply at each stage and in each scenario, relating the decision variables to the uncertain parameters realized in that scenario. They typically include resource constraints, demand satisfaction constraints, etc.
        *   **NACs:** These are *equality* constraints that ensure decisions are consistent across indistinguishable scenarios up to a given stage.  For example, if scenarios *s1* and *s2* are identical up to stage *t*, then `x_{t,s1} = x_{t,s2}`.
        *   **Dynamics/Recourse constraints:** These link decisions across stages within a single scenario, representing the evolution of the system. They describe how the state of the system changes from one stage to the next based on the decisions made and the realized uncertainties.

**How can we use it?**

1.  **Solve for Optimal Decisions:** By solving the deterministic equivalent using a standard optimization solver (e.g., Gurobi, CPLEX), we obtain the optimal decisions for each stage and scenario.
2.  **Analyze Sensitivity:** The deterministic equivalent allows us to analyze the sensitivity of the optimal solution to changes in scenario probabilities or uncertain parameters.
3.  **Evaluate the Value of Information:**  By comparing the expected value of the stochastic solution (obtained from the deterministic equivalent) to the value of a "wait-and-see" solution (where we solve the problem separately for each scenario after the uncertainty is revealed), we can quantify the value of information.
4.  **Implement Robust Strategies:** The solution to the deterministic equivalent provides a set of robust strategies that can be implemented under different realizations of uncertainty.

## 2) Application scenario

Consider a multi-period electricity generation planning problem. An electric utility company needs to decide how much electricity to generate from different sources (e.g., coal, gas, renewable energy) over the next 3 years.  The uncertain parameters are:

*   **Electricity demand:** Fluctuates each year and is influenced by weather patterns and economic activity. We represent this with a scenario tree showing potential demand levels each year.
*   **Fuel prices (e.g., gas prices):**  These can vary significantly and impact the cost of electricity generation.  Again, represented by a scenario tree.
*   **Renewable energy availability (e.g., solar and wind):** Varies based on weather conditions. Also represented by a scenario tree.

**The stochastic program would aim to minimize the expected cost of electricity generation over the 3 years, subject to the following constraints:**

*   **Demand satisfaction:**  The total electricity generated in each period must meet the demand in that period.
*   **Capacity constraints:** The amount of electricity generated from each source cannot exceed its capacity.
*   **Fuel availability constraints:** The amount of fuel consumed cannot exceed the available fuel supply.
*   **Non-anticipativity constraints:** The electricity generation decisions for the first year must be the same for all scenarios since we don't know which scenario will actually occur. The year 2 decisions must be the same for all scenarios that share the same first-year realization of demand, fuel prices, and renewable energy availability, and so on.

The Deterministic Equivalent would expand this problem by:

1.  Creating decision variables for the electricity generation from each source *in each year* *under each scenario*.
2.  Writing the objective function as the expected total cost of electricity generation, computed as the weighted sum of costs across all scenarios.
3.  Including all the demand satisfaction, capacity, and fuel availability constraints *for each year* *under each scenario*.
4.  Adding non-anticipativity constraints to ensure that generation decisions are identical for scenarios that are indistinguishable up to a certain stage. For example, the generation decisions for year 1 would be constrained to be equal across all scenarios.

Solving this deterministic equivalent provides the optimal generation plan for each year under each possible scenario, considering the uncertainty in demand, fuel prices, and renewable energy availability, and ensuring that decisions are made without looking into the future.

## 3) Python method (if possible)

```python
import pyomo.environ as pyo
import numpy as np

def create_deterministic_equivalent(scenario_tree, stage_vars, stage_constraints, objective_function):
    """
    Creates the deterministic equivalent formulation of a multistage stochastic program.

    Args:
        scenario_tree (dict): A dictionary representing the scenario tree.
                           Keys are node IDs, values are dictionaries containing:
                           'probability': Probability of the scenario.
                           'stage': Stage of the scenario.
                           'parent': ID of the parent node (None for the root).
                           'children': List of child node IDs.
        stage_vars (dict): A dictionary containing lists of pyomo variables for each stage.
                           Keys are stage numbers, values are lists of pyomo variables.
        stage_constraints (dict): A dictionary containing lists of pyomo constraints for each stage.
                               Keys are stage numbers, values are lists of pyomo constraints.
        objective_function (pyo.Expression): A Pyomo expression representing the stage-wise objective function.
                                 This should contain decision variables indexed by stage and scenario

    Returns:
        pyo.ConcreteModel: The deterministic equivalent Pyomo model.
    """

    model = pyo.ConcreteModel()

    # Nodes set
    model.NODES = set(scenario_tree.keys())

    # Stages set
    stages = set(scenario_tree[n]['stage'] for n in scenario_tree)
    model.STAGES = stages

    # 1. Variables (already defined using Pyomo)
    #    - Add the variables into the model (needed, even if defined externally)
    model.vars = pyo.Set(initialize=[v.name for stage in stage_vars.values() for v in stage])


    # 2. Objective Function
    def obj_rule(model):
      return sum(scenario_tree[node]['probability'] * objective_function.expr(model, node)
                 for node in model.NODES)

    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.MINIMIZE)  # Minimizing cost

    # 3. Constraints

    # Scenario-specific constraints
    model.scenario_constraints = pyo.ConstraintList()
    for node in model.NODES:
        stage = scenario_tree[node]['stage']
        for constr in stage_constraints.get(stage, []): # Check if the stage exist in the constraints
            model.scenario_constraints.add(constr.expr(model,node))

    # Non-Anticipativity Constraints (NACs)

    # Function to find scenarios with the same history
    def find_common_history(scenario_tree, stage):
        scenarios_by_history = {}
        for node, data in scenario_tree.items():
            if data['stage'] == stage:
                history = []
                current_node = node
                while scenario_tree[current_node]['parent'] is not None:
                    history.insert(0, current_node)  # Prepend to maintain order
                    current_node = scenario_tree[current_node]['parent']
                history_tuple = tuple(history)  # Tuples are hashable for dictionary keys
                if history_tuple not in scenarios_by_history:
                    scenarios_by_history[history_tuple] = []
                scenarios_by_history[history_tuple].append(node)
        return scenarios_by_history

    model.nacs = pyo.ConstraintList()
    for t in model.STAGES:
        if t > 0:  # NACs don't apply at the root stage (stage 0)
            common_histories = find_common_history(scenario_tree, t)
            for history, scenarios in common_histories.items():
                if len(scenarios) > 1:  # Only need NACs if there are multiple scenarios with the same history
                    #  Iterate over all variables defined in the previous stage
                    for var in stage_vars.get(t-1,[]): # check for empty dictionary

                      # We need to define some way to correctly access the variables of interest. Here is a dummy:
                      def access_var(model,scenario,var_name):
                          # Replace with an actual way to index the variables within your model.
                          # This depends heavily on how the scenario id is defined.
                          return getattr(model,var_name)[scenario] # Default scenario tree indexed as model.var[scenario]


                      for i in range(1, len(scenarios)):
                            # Add NAC to constrain decision variables for indistinguishable scenarios
                            model.nacs.add(access_var(model,scenarios[0],var.name) == access_var(model,scenarios[i],var.name)) # equality constraint

    return model

# Example Usage (Requires defining your scenario tree, stage_vars, stage_constraints, and objective_function appropriately)
# You need to replace the dummy functions and parameters with the actual variables
# and relationships relevant to your optimization model.

if __name__ == '__main__':

    # Dummy data for example
    scenario_tree = {
        0: {'probability': 1.0, 'stage': 0, 'parent': None, 'children': [1, 2]},
        1: {'probability': 0.5, 'stage': 1, 'parent': 0, 'children': [3, 4]},
        2: {'probability': 0.5, 'stage': 1, 'parent': 0, 'children': [5, 6]},
        3: {'probability': 0.25, 'stage': 2, 'parent': 1, 'children': []},
        4: {'probability': 0.25, 'stage': 2, 'parent': 1, 'children': []},
        5: {'probability': 0.25, 'stage': 2, 'parent': 2, 'children': []},
        6: {'probability': 0.25, 'stage': 2, 'parent': 2, 'children': []}
    }

    # Initialize variables
    stage_vars = {
        0: [pyo.Var(domain=pyo.NonNegativeReals, name='x0', initialize=1)],
        1: [pyo.Var(domain=pyo.NonNegativeReals, name='x1', initialize=1)],
        2: [pyo.Var(domain=pyo.NonNegativeReals, name='x2', initialize=1)],
    }

    # Constraints are defined externally. This is a dummy constraint.
    def constr_rule(model,node):
        stage = scenario_tree[node]['stage']
        if stage == 0:
            return stage_vars[0][0](model) >= 0
        elif stage == 1:
            return stage_vars[1][0](model) >= 0
        else:
            return stage_vars[2][0](model) >= 0


    stage_constraints = {
        0: [pyo.Constraint(rule=constr_rule, name='c0')],
        1: [pyo.Constraint(rule=constr_rule, name='c1')],
        2: [pyo.Constraint(rule=constr_rule, name='c2')]
    }


    # Objective function (example)
    class Objective_Expr:
        def expr(self, model, node):
            stage = scenario_tree[node]['stage']
            # In reality, you would access the specific variable correctly from your model.
            if stage == 0:
                return stage_vars[0][0](model) * 1
            elif stage == 1:
                return stage_vars[1][0](model) * 2
            else:
                return stage_vars[2][0](model) * 3


    objective_function = Objective_Expr()

    # Create the deterministic equivalent model
    deterministic_model = create_deterministic_equivalent(scenario_tree, stage_vars, stage_constraints, objective_function)

    # Solve the model
    solver = pyo.SolverFactory('glpk')  # or 'gurobi', 'cplex', etc.
    solver.solve(deterministic_model)

    # Print the solution (example)
    for var_name in deterministic_model.vars:
        for node in deterministic_model.NODES:
            try:
                print(f"{var_name}[{node}]: {deterministic_model.find_component(var_name)[node].value}")
            except:
                pass
```

**Explanation:**

1.  **`create_deterministic_equivalent(scenario_tree, stage_vars, stage_constraints, objective_function)` Function:**
    *   Takes the scenario tree, stage variables, stage constraints, and objective expression as input.
    *   Creates a `pyo.ConcreteModel` to represent the deterministic equivalent.
    *   Iterates through the scenario tree, adding the stage-specific constraints for each scenario.
    *   Implements the Non-Anticipativity Constraints (NACs) by:
        *   Finding scenarios that share the same history up to a given stage using function `find_common_history`.
        *   For each set of scenarios with a common history, iterates through the variables and enforces equality constraints between the corresponding decision variables in those scenarios.  The essential part here is correctly accessing the variables using a model index.
    *   Defines the objective function as the expected value across all scenarios.
    *   Returns the deterministic equivalent Pyomo model.
2.  **`scenario_tree` Dictionary:** This represents the tree structure.  Each key is a node ID, and the value is a dictionary containing the node's probability, stage, parent node, and child nodes.
3.  **`stage_vars` Dictionary:** Stores the Pyomo variables for each stage.  This example creates simple scalar variables indexed by stage. **Crucially, your implementation needs variables indexed by *both* stage and scenario/node to represent decisions within the scenario tree.**
4.  **`stage_constraints` Dictionary:** Holds the Pyomo constraints defined for each stage. This example has dummy constraints that simply bound x >=0
5.  **`Objective_Expr` Class:** Defines the objective function as a Pyomo expression, accessed through the objective_function.expr(model,node) call.  This returns an objective expression.
6.  **Example Usage:**
    *   Creates dummy data for the scenario tree, stage variables, and stage constraints.
    *   Calls the `create_deterministic_equivalent` function to construct the deterministic equivalent model.
    *   Solves the model using a solver (GLPK in this example). You may need to install an appropriate solver.
    *   Prints the solution.

**Important Notes:**

*   **Variable Indexing:**  The example code uses simplified scalar variables.  In a real application, you will need to create decision variables indexed by *both* stage and scenario. For instance, `model.x = pyo.Var(model.STAGES, model.NODES, domain=pyo.NonNegativeReals)` and properly adjust the `access_var` function.
*   **Constraint Definition:** The `stage_constraints` dictionary and the `constr_rule` function are placeholders. You'll need to define your actual constraints based on the specific problem you're modeling. They must be indexed correctly (e.g., by stage and scenario).
*   **Objective Function Definition:** Replace the dummy objective function definition with the actual objective function for your problem.  It also needs to correctly access variables indexed by stage and scenario.
*   **Scenario Tree Representation:** Ensure the scenario tree representation is accurate and captures the dependencies between scenarios.
*   **Solver Selection:**  Choose an appropriate solver (e.g., Gurobi, CPLEX) based on the size and complexity of the deterministic equivalent problem.  You might need to install the solver and ensure it is accessible from Pyomo.

This code provides a template. Adapting it to your specific problem requires careful consideration of how you define your decision variables, constraints, and objective function within the scenario tree structure.  The `access_var` function is particularly important, and it needs to be customized based on the specific indexing scheme of your Pyomo variables. The example uses string variables as an alternative, it still needs to be adjusted to a custom indexing scheme.

## 4) Follow-up question

How does the size of the deterministic equivalent problem grow as the number of stages and the branching factor of the scenario tree increase? What are some techniques to mitigate the computational challenges associated with solving very large deterministic equivalents (e.g., decomposition methods)?
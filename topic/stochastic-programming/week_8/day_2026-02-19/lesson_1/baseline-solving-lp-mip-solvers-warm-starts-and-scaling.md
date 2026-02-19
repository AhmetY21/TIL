---
title: "Baseline Solving: LP/MIP solvers, warm starts, and scaling"
date: "2026-02-19"
week: 8
lesson: 1
slug: "baseline-solving-lp-mip-solvers-warm-starts-and-scaling"
---

# Topic: Baseline Solving: LP/MIP solvers, warm starts, and scaling

## 1) Formal definition (what is it, and how can we use it?)

**Baseline Solving** in the context of Stochastic Programming refers to establishing a basic, initial solution to a deterministic equivalent problem (DEP) or a sample average approximation (SAA) of the stochastic problem. This initial solution is obtained by directly solving the problem using standard LP/MIP solvers without employing advanced decomposition or approximation techniques specifically designed for stochastic optimization.  Baseline solving serves as a performance benchmark against which more sophisticated methods can be evaluated.

*   **LP/MIP Solvers:** Linear Programming (LP) and Mixed-Integer Programming (MIP) solvers are software packages designed to find optimal solutions for linear optimization problems (with and without integer variables, respectively).  They are the workhorses of optimization and typically employ branch-and-bound, cutting planes, and other algorithms.  Popular solvers include Gurobi, CPLEX, and open-source options like GLPK or CBC.
*   **Warm Starts:** A warm start provides the solver with an initial feasible (or near-feasible) solution to the problem. This can significantly reduce the solving time, especially for problems that are structurally similar to previously solved problems.  The solver uses this initial solution as a starting point and refines it to find the optimal solution more quickly.
*   **Scaling:** Scaling refers to pre-processing techniques that improve the numerical properties of the optimization problem. This involves scaling variables and constraints to have values within a more manageable range, which can prevent numerical instability and improve solver performance. Poorly scaled problems can lead to inaccurate solutions or even solver failures.

**How to use it:**

1.  **Model the Problem:** Formulate the stochastic problem either as a deterministic equivalent (DEP) or a sample average approximation (SAA). The DEP creates a large, deterministic LP/MIP. The SAA approximates the probability distribution with a finite set of scenarios and constructs a corresponding LP/MIP.
2.  **Solve with LP/MIP Solver:** Use a standard LP/MIP solver to find the optimal solution to the DEP/SAA.
3.  **Implement Warm Starts:** If a feasible solution is available (e.g., from a previous iteration of an algorithm or from solving a simplified version of the problem), pass it to the solver as a warm start.
4.  **Scale the Problem:** Before solving, apply scaling techniques to ensure that the coefficients and variable bounds are well-conditioned.
5.  **Benchmark:** Use the solution obtained as a baseline to compare the performance of more advanced stochastic programming techniques.

## 2) Application scenario

**Inventory Management under Uncertainty:**

Consider an inventory management problem where a retailer needs to decide how many units of a product to order in each period to meet uncertain demand. The demand in each period is a random variable.  The retailer wants to minimize the total cost, which includes ordering costs, holding costs, and shortage costs.

1.  **SAA Formulation:** We can approximate the stochastic demand with a set of scenarios (e.g., historical data or simulation outputs).
2.  **Baseline Solution:**  We can solve the resulting SAA problem as a large-scale LP using a solver like Gurobi. This provides an initial inventory ordering policy.
3.  **Warm Start:**  If the retailer solves this problem repeatedly with slightly different scenarios (e.g., rolling horizon planning), they can use the solution from the previous period as a warm start for the current period's problem.
4.  **Scaling:**  If the ordering costs are significantly smaller than the holding costs, scaling the problem can improve the solver's numerical stability.  For example, divide the ordering cost coefficients by a suitable constant.

## 3) Python method (if possible)

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_inventory_saa(demands, ordering_cost, holding_cost, shortage_cost, initial_inventory=0):
    """
    Solves a sample average approximation (SAA) of an inventory management problem
    using Gurobi.

    Args:
        demands (list of lists): A list of scenarios, where each scenario is a list of demands over time.
        ordering_cost (float): Cost per unit ordered.
        holding_cost (float): Cost per unit held in inventory.
        shortage_cost (float): Cost per unit of unmet demand.
        initial_inventory (float): Initial inventory level.

    Returns:
        dict: A dictionary containing the optimal ordering quantities and inventory levels for each scenario.
    """
    num_scenarios = len(demands)
    num_periods = len(demands[0])

    try:
        # Create a new Gurobi model
        model = gp.Model("inventory_saa")

        # Create variables
        order_quantity = model.addVars(num_scenarios, num_periods, vtype=GRB.CONTINUOUS, name="order_quantity", lb=0)
        inventory_level = model.addVars(num_scenarios, num_periods, vtype=GRB.CONTINUOUS, name="inventory_level", lb=0)
        shortage = model.addVars(num_scenarios, num_periods, vtype=GRB.CONTINUOUS, name="shortage", lb=0)

        # Set objective function (minimize total cost)
        objective = gp.quicksum(
            (ordering_cost * order_quantity[s, t] +
             holding_cost * inventory_level[s, t] +
             shortage_cost * shortage[s, t])
            for s in range(num_scenarios) for t in range(num_periods)
        ) / num_scenarios  # Average cost across scenarios
        model.setObjective(objective, GRB.MINIMIZE)

        # Add constraints (inventory balance)
        for s in range(num_scenarios):
            # First period inventory
            model.addConstr(inventory_level[s, 0] == initial_inventory + order_quantity[s, 0] - demands[s][0] + shortage[s,0], name=f"inventory_balance_{s}_0")
            for t in range(1, num_periods):
                model.addConstr(inventory_level[s, t] == inventory_level[s, t - 1] + order_quantity[s, t] - demands[s][t] + shortage[s, t], name=f"inventory_balance_{s}_{t}")

        # Optimize model
        model.optimize()

        # Store results
        results = {}
        results["order_quantity"] = { (s,t): order_quantity[s, t].x for s in range(num_scenarios) for t in range(num_periods) }
        results["inventory_level"] = { (s,t): inventory_level[s, t].x for s in range(num_scenarios) for t in range(num_periods) }
        results["shortage"] = { (s,t): shortage[s, t].x for s in range(num_scenarios) for t in range(num_periods) }
        results["objective_value"] = model.objVal

        return results

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')


# Example usage:
demands = [[10, 12, 8], [12, 15, 10], [8, 10, 6]]  # 3 scenarios, 3 periods
ordering_cost = 2
holding_cost = 1
shortage_cost = 5
initial_inventory = 5

results = solve_inventory_saa(demands, ordering_cost, holding_cost, shortage_cost, initial_inventory)

if results:
  print("Optimal objective value:", results["objective_value"])
  print("Optimal order quantities:", results["order_quantity"])
  print("Optimal inventory levels:", results["inventory_level"])
  print("Optimal shortages:", results["shortage"])


# Example of warm starting (not implemented fully, only shown conceptually)
# In a real application, you would pass the solution from a previous run as the start value.
# model.start = previous_solution
# model.optimize()

```

**Explanation:**

1.  **`solve_inventory_saa` Function:** This function takes the demand scenarios, costs, and initial inventory as input.
2.  **Gurobi Model:** It creates a Gurobi model (`model = gp.Model("inventory_saa")`).
3.  **Variables:** It defines variables for order quantity, inventory level, and shortage for each scenario and period.
4.  **Objective Function:** It sets the objective function to minimize the average total cost across all scenarios.
5.  **Constraints:** It adds constraints to enforce the inventory balance equation for each scenario and period.  The `model.addConstr` method adds linear constraints.
6.  **Optimization:** `model.optimize()` solves the LP.
7.  **Results:**  It extracts the optimal solution and returns it in a dictionary.
8.  **Example Usage:** The code provides an example of how to use the function with sample demand data.

**Warm Start (Conceptual):**  The commented-out section demonstrates the idea of a warm start. In a real application, you would get the solution from a previous run (or a heuristic solution) and set the `start` attribute of the variables before calling `model.optimize()` again.  This tells Gurobi to use that initial solution as a starting point.

**Scaling:**  Scaling isn't explicitly done in this code.  However, you could add scaling by multiplying or dividing variables and coefficients by appropriate constants before creating the Gurobi model. This would be most important if the ordering, holding, and shortage costs have vastly different magnitudes.  For example, if the shortage cost is 1000 times larger than the ordering cost, divide the shortage cost coefficients by 1000 (and the shortage variables by 1000) before creating the model.  Remember to scale back the solution after solving.

## 4) Follow-up question

How can scenario reduction techniques (e.g., scenario clustering or sampling methods) be integrated with baseline solving to improve the computational efficiency of solving the stochastic program, and how would these techniques impact the quality of the baseline solution compared to using all scenarios?
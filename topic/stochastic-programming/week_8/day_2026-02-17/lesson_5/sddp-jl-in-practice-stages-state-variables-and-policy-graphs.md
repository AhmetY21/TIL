---
title: "SDDP.jl in Practice: stages, state variables, and policy graphs"
date: "2026-02-17"
week: 8
lesson: 5
slug: "sddp-jl-in-practice-stages-state-variables-and-policy-graphs"
---

# Topic: SDDP.jl in Practice: stages, state variables, and policy graphs

## 1) Formal definition (what is it, and how can we use it?)

SDDP.jl is a Julia package for solving multistage stochastic linear programs using Stochastic Dual Dynamic Programming (SDDP). When applying SDDP.jl, understanding the concepts of stages, state variables, and policy graphs is crucial for formulating and solving problems effectively.

*   **Stages:** A stage represents a point in time where a decision needs to be made. SDDP explicitly models the problem as a sequence of these stages. Decisions made at one stage influence the future state and are made under uncertainty, which is resolved stage-by-stage. The problem is "decomposed" into these stages. The goal is to find a sequence of decisions across all stages that minimizes the expected cost over the planning horizon.

*   **State Variables:** State variables represent the information that carries over from one stage to the next. They link the stages together, reflecting the impact of past decisions and realized uncertainties on the current stage's optimization problem. They encapsulate the relevant aspects of the past that affect future decisions. Examples include inventory levels, reservoir water levels, or capital stock. Each stage takes the current state as input and outputs an updated state that is passed on to the subsequent stage. SDDP.jl manages the backward induction through these state variables.

*   **Policy Graphs (Implicit):** The policy graph isn't a data structure explicitly managed by the user in SDDP.jl, but it's the *conceptual* representation of the decision-making process learned by the algorithm. It implicitly represents the optimal policy at each stage based on the possible states. SDDP builds up the *cut representation* of the cost-to-go function.  In essence, the algorithm solves the problem backward, starting from the last stage and working its way back to the first, approximating the optimal value function (the cost-to-go function) for each possible state at each stage. This value function is approximated by a series of linear cuts. The optimal policy at any given stage is then derived from this approximated value function and the current state. While not explicitly visualized by SDDP.jl in standard usage, understanding this implicit policy graph is fundamental to interpreting the results. Each cut in the cost-to-go function represents an extreme point solution that may be optimal for a given state. Thus, given a state, you can evaluate the different cuts to determine the optimal action.

In practice, to use SDDP.jl, you define the stage-wise optimization problems, specifying the objective function, constraints, and how state variables evolve between stages. SDDP.jl then iteratively samples scenarios and generates cuts (linear approximations of the value function) to approximate the optimal policy.

## 2) Application scenario

Consider a water reservoir management problem over a 5-year planning horizon (5 stages).  At the beginning of each year (stage), a decision must be made about how much water to release for irrigation and power generation, considering uncertain inflow (rainfall) into the reservoir.

*   **Stages:** Each year represents a stage (stage 1 to stage 5).
*   **State Variable:** The water level of the reservoir at the beginning of each year is the state variable. It's influenced by the previous year's decisions (release amount), rainfall (uncertainty), and evaporation (deterministic).
*   **Policy Graph (Implicit):** The algorithm learns the optimal water release policy for each stage depending on the current reservoir water level.  For example, if the reservoir is full, the policy might dictate releasing a large amount to avoid spillage and potential flood damage.  If the reservoir is low, the policy might restrict releases to conserve water for future use. The learned value functions at each stage capture this policy graph.

## 3) Python method (if possible)

While SDDP.jl is a Julia package, it's often used in conjunction with other tools, and its results can certainly be consumed by Python. Here's how you could *simulate* or *evaluate* a policy learned by SDDP.jl *within* Python, assuming you have the cuts (value function approximations) exported from the Julia SDDP.jl model. This shows how you *use* the result of SDDP, not how you *run* SDDP from Python. Running SDDP from Python requires interfacing with Julia.

First, we assume that somehow you have managed to transfer the cuts computed by SDDP.jl to Python. This involves reading them from a file, or via a shared database, or an API. We assume that the cuts are in a `cuts` variable, formatted like a list of (alpha, beta) pairs for each stage. `alpha` is a vector of coefficients for the state variable, and `beta` is a scalar intercept for the cut.

```python
import numpy as np

def evaluate_policy(cuts, initial_state, inflows):
    """
    Simulates the reservoir management policy over time, given the SDDP cuts.

    Args:
        cuts: A list of lists of (alpha, beta) pairs for each stage.
              Each inner list represents the cuts for a particular stage.
              alpha is a numpy array, beta is a scalar.
        initial_state: The initial water level of the reservoir.
        inflows: A list of inflows (rainfall) for each year (stage).

    Returns:
        A list of reservoir water levels at the beginning of each stage,
        a list of release amounts for each stage, and a list of costs for each stage.
    """

    state = initial_state
    states = [state]
    releases = []
    costs = []
    num_stages = len(cuts)

    for stage in range(num_stages):
        stage_cuts = cuts[stage]

        # 1. Determine the optimal release amount by minimizing the cost-to-go function + current cost
        best_release = None
        min_cost = float('inf')

        # Simple linear program to determine the optimal release. This is a naive implementation.
        # In a real use case, you'd replace this with a proper LP solver like `scipy.optimize.linprog`
        # that incorporates release bounds, power generation costs, etc.
        # This is for demonstration purposes only.

        for release in np.linspace(0, state, 100):  # Try 100 different release amounts
            current_cost = release  # Simple cost function: cost is equal to release amount.

            # Evaluate the cost-to-go function (value function) for this release.
            cost_to_go = float('inf') # Start assuming infinite cost-to-go
            for alpha, beta in stage_cuts:
              potential_cost_to_go = alpha @ np.array([state - release + inflows[stage]]) + beta
              cost_to_go = min(cost_to_go, potential_cost_to_go) # Keep the minimum

            total_cost = current_cost + cost_to_go

            if total_cost < min_cost:
                min_cost = total_cost
                best_release = release
        #print(f"Stage {stage+1}: state {state}, best release {best_release}, min_cost {min_cost}")
        # 2. Update the state based on the release and the inflow
        release = best_release #take the best release from the optimization
        releases.append(release)
        costs.append(min_cost)  #The cost including the cost to go is the "minimum total cost"

        state = state - release + inflows[stage]  # State transition
        states.append(state) #Add next state to list of states

    return states, releases, costs

# Example Usage (assuming you have `cuts` loaded from SDDP.jl):
# Note that this is a highly simplified and illustrative example.
# A real application would involve:
#   - Reading cuts from a file (e.g., CSV, JSON)
#   - More complex stage-wise optimization within the simulation loop
#   - Realistic cost functions, constraints, and inflow scenarios.

#Dummy Cuts data (alpha, beta)
dummy_cuts = [
    [ (np.array([0.5]), 10.0), (np.array([0.7]), 12.0) ],  # Stage 1 cuts
    [ (np.array([0.6]), 8.0), (np.array([0.8]), 11.0) ],   # Stage 2 cuts
    [ (np.array([0.4]), 9.0), (np.array([0.9]), 13.0) ],  # Stage 3 cuts
    [ (np.array([0.3]), 7.0), (np.array([0.5]), 10.0) ],   # Stage 4 cuts
    [ (np.array([0.2]), 6.0), (np.array([0.4]), 9.0) ]    # Stage 5 cuts
]

initial_reservoir_level = 50.0
inflow_scenarios = [10.0, 12.0, 8.0, 15.0, 5.0]

reservoir_levels, release_amounts, total_costs = evaluate_policy(dummy_cuts, initial_reservoir_level, inflow_scenarios)

print("Reservoir Levels:", reservoir_levels)
print("Release Amounts:", release_amounts)
print("Total Costs:", total_costs)
```

This code *simulates* how the system would evolve given a policy learned by SDDP.jl and the given inflow scenarios. The key point is that the simulation *uses* the output (cuts) from SDDP.jl which has been used to approximate the policy graph.

**Important Notes:**

*   The Python code provided is a **simplified illustration**. It does *not* run SDDP.jl itself.  It assumes you have already solved the problem with SDDP.jl and obtained the cuts.
*   In practice, transferring the cuts from Julia to Python requires a suitable data format (e.g., CSV, JSON, Arrow) and a method to read that data in Python. The example uses manually specified `dummy_cuts`.
*   The stage-wise optimization within the simulation loop would typically involve a proper linear programming solver (`scipy.optimize.linprog`) to handle constraints and costs more realistically. The example just iterates over possible releases.
*   The complexity of the simulation depends heavily on the complexity of the underlying system you are modeling.

## 4) Follow-up question

How does the choice of the state variable definition influence the performance and accuracy of the SDDP.jl algorithm? Specifically, can you describe how adding or removing state variables affects the computational burden and the quality of the resulting policy?
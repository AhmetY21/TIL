---
title: "DoWhy Workflow: Model → Identify → Estimate → Refute"
date: "2026-02-13"
week: 7
lesson: 6
slug: "dowhy-workflow-model-identify-estimate-refute"
---

# Topic: DoWhy Workflow: Model → Identify → Estimate → Refute

## 1) Formal definition (what is it, and how can we use it?)

The DoWhy workflow, comprising the steps **Model → Identify → Estimate → Refute**, is a structured approach to causal inference using the DoWhy Python library. It aims to provide a principled and robust way to estimate causal effects from observational data. Let's break down each step:

*   **Model (Causal Graph):** This step involves explicitly representing your causal assumptions in the form of a causal graph (Directed Acyclic Graph, or DAG).  The DAG visualizes the relationships between variables, including which variables are assumed to cause other variables.  This step forces you to articulate your beliefs about how the world works, even before looking at the data. A correct causal graph is crucial for valid causal inference. If the graph is incorrect, downstream steps can produce biased or misleading results.  You use domain expertise and prior knowledge to build this graph.

*   **Identify:** This step uses the causal graph from the Model step to algorithmically determine if the causal effect you are interested in is *identifiable*. Identifiability means that it is theoretically possible to estimate the causal effect from the available data, given the assumptions encoded in your graph. DoWhy leverages graph-based causal inference methods (e.g., back-door criterion, front-door criterion) to find valid adjustment sets or instrumental variables. The output of this step is an *identification query* which outlines what statistical estimand(s) (e.g., P(Y|do(X))) can be estimated from the observational data. If the effect is not identifiable based on your graph, you need to revise your graph or consider collecting different data.

*   **Estimate:** This step takes the identification query from the Identify step and uses statistical methods to estimate the causal effect. DoWhy provides various estimation methods, including regression, matching, inverse probability weighting (IPW), and instrumental variables (IV) regression. The choice of estimator depends on the identification strategy and the nature of the data. This step yields a concrete estimate of the causal effect, along with uncertainty estimates (e.g., confidence intervals).

*   **Refute:**  This step critically examines the robustness of the estimated causal effect by performing sensitivity analyses.  The goal is to assess how sensitive the estimated effect is to violations of the assumptions made in the earlier steps. Refutation methods test the validity of the assumptions underlying the identification strategy. For example, one refutation technique might add unobserved confounders to the model and see how much they would need to influence the relationship between treatment and outcome to invalidate the findings. Other refutation methods test the sensitivity of the estimate to variations in the treatment assignment mechanism. If the estimated effect is sensitive to plausible violations of the assumptions, the user should be more cautious in interpreting the results.

By following this workflow, you increase the confidence in your causal inferences and make your reasoning more transparent and reproducible.

## 2) Application scenario

Imagine you want to understand the causal effect of a new job training program (treatment, `X`) on participants' income (outcome, `Y`). You have observational data on individuals who participated in the program and those who didn't. However, participants who enrolled in the program may differ systematically from non-participants (e.g., motivated people may be more likely to self-select into the program). This is a confounding variable (`C`, motivation). Also, years of experience might impact both the training program acceptance, and the subsequent income (`Z`, experience).

1.  **Model:** You create a causal graph showing that:
    *   `X` (Training) causes `Y` (Income)
    *   `C` (Motivation) causes both `X` (Training) and `Y` (Income)
    *   `Z` (Experience) causes both `X` (Training) and `Y` (Income)
2.  **Identify:**  DoWhy uses your causal graph to identify that you can estimate the causal effect of training on income by adjusting for `C` and `Z` (backdoor criterion).
3.  **Estimate:** You use a regression model to estimate the causal effect of training on income, controlling for `C` and `Z`.
4.  **Refute:**  You perform a sensitivity analysis to check whether the estimated effect is robust to unobserved confounders that might be correlated with both training and income. For example, you might add a hypothetical unobserved confounder representing "access to resources" and examine how much the observed effect changes.

## 3) Python method (if possible)

```python
import dowhy
from dowhy import CausalModel
import pandas as pd

# 1. Model: Define the causal graph
# Assuming df is a Pandas DataFrame with columns 'training', 'income', 'motivation', 'experience'
graph = """
digraph {
training[label="Training"];
income[label="Income"];
motivation[label="Motivation"];
experience[label="Experience"];
motivation -> training;
motivation -> income;
experience -> training;
experience -> income;
training -> income;
}
"""

# Sample dataframe
data = {'training': [0, 1, 0, 1, 0],
        'income': [20000, 40000, 25000, 50000, 30000],
        'motivation': [3, 7, 4, 8, 5],
        'experience': [2, 6, 3, 7, 4]}
df = pd.DataFrame(data)


# Create a causal model
model = CausalModel(
    data=df,
    graph=graph.replace("\n", " "),  # DoWhy needs graph in a single line
    treatment="training",
    outcome="income"
)

# 2. Identify: Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)

# 3. Estimate: Estimate the causal effect using a suitable method (e.g., regression)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    control_value=0,
    treatment_value=1,
)
print(estimate)

# Get the estimated effect, confidence intervals, and more
print("Causal Estimate is " + str(estimate.value))

# 4. Refute: Refute the obtained estimate using various methods
refute_results = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause"  # Test for unobserved confounders
)
print(refute_results)


refute_results_placebo = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter", # Add random treatment
    placebo_type="binary"
)

print(refute_results_placebo)

```

## 4) Follow-up question

How do I choose the most appropriate refutation method for a given causal inference problem? What factors should I consider?
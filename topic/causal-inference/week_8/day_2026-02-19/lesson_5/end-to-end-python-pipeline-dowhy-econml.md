---
title: "End-to-End Python Pipeline (DoWhy + EconML)"
date: "2026-02-19"
week: 8
lesson: 5
slug: "end-to-end-python-pipeline-dowhy-econml"
---

# Topic: End-to-End Python Pipeline (DoWhy + EconML)

## 1) Formal definition (what is it, and how can we use it?)

An end-to-end Python pipeline using DoWhy and EconML streamlines the entire causal inference process, from problem framing and identification to estimation and sensitivity analysis.

*   **DoWhy:**  Focuses on the identification step, explicitly modeling causal assumptions and allowing you to rigorously test whether you can estimate the causal effect you're interested in based on the available data and assumed causal structure. It uses the "Causal Inference: The Book of Why" framework. It lets you:

    *   Create a causal graph (DAG - Directed Acyclic Graph) representing your assumptions about the causal relationships between variables.
    *   Identify the causal effect of a treatment variable on an outcome variable using back-door, front-door, and instrumental variable methods.
    *   Test the robustness of your causal estimates through sensitivity analysis.

*   **EconML:** Specializes in estimating heterogeneous treatment effects, i.e., how the causal effect varies across different subgroups of the population.  It provides a suite of machine learning methods designed to handle the challenges of causal inference, such as confounding and selection bias, in observational data. Key Features include:

    *   **Meta-Learners:**  S-Learner, T-Learner, X-Learner that combine machine learning models to estimate the treatment effect.
    *   **Double Machine Learning:** Estimators that leverage orthogonalization techniques to reduce bias.
    *   **Instrumental Variable Methods:** Methods designed for settings where you have an instrument to address endogeneity.

The end-to-end pipeline enables a principled approach to causal inference:

1.  **Problem Definition & Causal Graph Construction (DoWhy):**  Define the research question, identify the treatment, outcome, and potential confounders, and represent these relationships in a causal graph (DAG).

2.  **Identification (DoWhy):** Use DoWhy to determine if the desired causal effect is identifiable given the DAG and available data.  This involves finding a valid adjustment set (e.g., using the back-door criterion) or using front-door identification.

3.  **Estimation (EconML):**  Use EconML to estimate the average treatment effect (ATE) or conditional average treatment effect (CATE) based on the identified causal effect and the chosen machine learning method. Consider effect heterogeneity.

4.  **Refutation (DoWhy):** Perform robustness checks and sensitivity analysis using DoWhy to assess how sensitive the estimates are to violations of the causal assumptions (e.g., unobserved confounders).

We can use it to rigorously and reproducibly conduct causal inference on observational data, making it applicable in various fields such as economics, marketing, healthcare, and policy evaluation.

## 2) Application scenario

**Scenario:**  A company wants to understand the impact of a new marketing campaign (the treatment) on sales (the outcome).  They have observational data that includes customer demographics, past purchase history, and whether or not each customer was exposed to the campaign.

**Challenge:**  Customers who are more likely to buy the product *anyway* might be more likely to be targeted by the marketing campaign. This is confounding. There are probably unobserved confounders too (e.g., word-of-mouth marketing effectiveness).

**Pipeline Steps:**

1.  **Causal Graph:** Create a DAG to represent the relationships between campaign exposure, sales, customer demographics, purchase history, and potential unobserved confounders.

2.  **Identification:** Use DoWhy to determine if the effect of the campaign on sales can be identified using the back-door criterion. We may need to adjust for customer demographics and purchase history.

3.  **Estimation:** Use EconML to estimate the average treatment effect (ATE) of the campaign on sales.  Furthermore, we can estimate the CATE to understand which customer segments respond best to the campaign (e.g., younger vs. older customers).

4.  **Refutation:**  Conduct sensitivity analysis to assess how the estimated effect changes if there is an unobserved confounder that influences both campaign exposure and sales.

## 3) Python method (if possible)

```python
import dowhy
from dowhy import CausalModel
import econml
from econml.dml import LinearDML
import pandas as pd
import numpy as np

# Generate some synthetic data
np.random.seed(42)
n_samples = 500
data = pd.DataFrame({
    'age': np.random.randint(20, 60, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'campaign': np.random.binomial(1, 0.5, n_samples),  # Treatment: 1 if exposed to campaign, 0 otherwise
    'sales': 0 # Initialize sales
})

# Simulate sales based on age, income, campaign, and a hidden confounder (simulated directly)
hidden_confounder = np.random.normal(0, 1, n_samples)
data['sales'] = 100 + 0.5 * data['age'] + 0.001 * data['income'] + 50 * data['campaign'] + 20 * hidden_confounder + np.random.normal(0, 5, n_samples)

# 1. Causal Model
model = CausalModel(
    data=data,
    treatment='campaign',
    outcome='sales',
    graph="""digraph {
        age -> sales;
        income -> sales;
        campaign -> sales;
        age -> campaign;
        income -> campaign;
        U -> campaign;
        U -> sales;
    }""",
    common_causes=['age', 'income'],
    instruments=None,
    effect_modifiers=['age', 'income']
)

# 2. Identification
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)


# 3. Estimation
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression",
                                 control_value=0,
                                 treatment_value=1)
print(estimate)

# Estimate heterogeneous treatment effects using EconML
# Prepare data
Y = data['sales'].values
T = data['campaign'].values
X = data[['age', 'income']].values
W = data[['age', 'income']].values # Confounders to adjust for

# Linear DML model (a common and relatively simple choice)
dml_model = LinearDML(random_state=42, discrete_treatment=True)
dml_model.fit(Y, T, X=X, W=W)

# Estimate the ATE
ate = dml_model.ate(X)
print(f"ATE from EconML: {ate}")

# Estimate the CATE (effect modifiers are 'age' and 'income')
cate = dml_model.effect(X) # Returns a vector of CATEs for each sample
print(f"CATE for the first 5 samples: {cate[:5]}")

# 4. Refutation (Example: Add Random Common Cause)
res_random = model.refute_estimate(estimate, method_name="random_common_cause")
print(res_random)


# Note: More sophisticated refutation methods exist in DoWhy.  This is just an example.
```

## 4) Follow-up question

How can we handle situations where the causal graph is unknown or partially known? Can we use causal discovery algorithms within the DoWhy + EconML framework to learn the graph structure from the data and then proceed with causal inference? If so, are there specific considerations or limitations we should be aware of when using automatically learned causal graphs?
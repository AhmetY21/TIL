---
title: "Capstone Project Ideas (Pick one design + one library)"
date: "2026-02-20"
week: 8
lesson: 2
slug: "capstone-project-ideas-pick-one-design-one-library"
---

# Topic: Capstone Project Ideas (Pick one design + one library)

## 1) Formal definition (what is it, and how can we use it?)

This topic concerns the generation and selection of feasible and impactful capstone project ideas in causal inference, specifically focusing on pairing a study design with a relevant Python library. The 'design' refers to the method you employ to establish causality (e.g., A/B testing, Regression Discontinuity, Instrumental Variables). The 'library' refers to the Python package you'll use to implement your chosen design and analyze the data (e.g., `DoWhy`, `EconML`, `CausalImpact`).

The goal is to create a project that:

*   **Addresses a concrete problem:** The project should be grounded in a real-world scenario where identifying causal relationships can lead to actionable insights.
*   **Demonstrates mastery of causal inference principles:** The project should showcase your understanding of the chosen design and its underlying assumptions.
*   **Employs appropriate analytical techniques:** The project should utilize the selected library effectively to estimate causal effects and assess the robustness of the findings.
*   **Offers interpretable results:** The project should clearly communicate the estimated causal effects and their implications for decision-making.

By pairing a specific design with a library, you can streamline the project scope and focus on implementing the chosen causal inference method efficiently.  We can use this approach to:

*   Identify appropriate designs for answering specific causal questions.
*   Develop proficiency in using Python libraries for causal inference.
*   Build a portfolio of projects that demonstrate expertise in causal inference.

## 2) Application scenario

Let's consider the scenario of evaluating the impact of a new personalized learning platform on student performance. A school district implements the platform in some schools but not others. We want to estimate the causal effect of using the platform on student test scores.

Here's how we could approach this with different designs and libraries:

*   **Design: Difference-in-Differences (DID)**
    *   Scenario: Compare the change in test scores for students in schools that adopted the platform (treatment group) versus students in schools that didn't (control group), before and after the platform was implemented.  This helps control for pre-existing differences between the schools.

*   **Design: Propensity Score Matching (PSM)**
    *   Scenario: Since the adoption of the platform wasn't random, we use PSM to create comparable treatment and control groups based on pre-treatment characteristics (e.g., prior test scores, demographics, school resources).

*   **Design: Regression Discontinuity (RD)**
    *   Scenario:  Suppose schools were selected for the platform based on a cutoff score on a needs assessment. We can use RD to estimate the causal effect by comparing outcomes for schools just above and below the cutoff.

Here's how the library selection would look, paired with the designs above:

*   **DID + `statsmodels`:** `statsmodels` provides tools for running regression models with interaction terms to estimate the DID effect.

*   **PSM + `scikit-learn` + `causalml`:** `scikit-learn` can be used to estimate the propensity scores, and `causalml` provides tools to further analyze causal effects within the matched groups.

*   **RD + `statsmodels` / `rdd` (R in Python):** `statsmodels` can be used to run regressions within a bandwidth around the cutoff or you can use the `rdd` package which focuses specifically on regression discontinuity designs (this usually involves installing R and using `rpy2`).

## 3) Python method (if possible)

Here's an example of using Difference-in-Differences with `statsmodels`:

```python
import pandas as pd
import statsmodels.formula.api as smf

# Sample data (replace with your actual data)
data = {'school_id': [1, 1, 1, 1, 2, 2, 2, 2],
        'year': [2020, 2021, 2020, 2021, 2020, 2021, 2020, 2021],
        'treated': [0, 0, 0, 0, 1, 1, 1, 1],
        'test_score': [70, 72, 75, 77, 65, 70, 72, 78]}
df = pd.DataFrame(data)

# Create an interaction term for treated * post_treatment
df['post_treatment'] = df['year'].apply(lambda x: 1 if x == 2021 else 0)
df['interaction'] = df['treated'] * df['post_treatment']

# Run the regression
model = smf.ols("test_score ~ treated + post_treatment + interaction", data=df)
results = model.fit()

# Print the results
print(results.summary())

# The coefficient of 'interaction' is the DID estimate of the treatment effect
```

In this example, the `interaction` term's coefficient estimates the impact of the treatment (personalized learning platform) on test scores. A positive and statistically significant coefficient suggests a positive effect.

## 4) Follow-up question

How do you handle unobserved confounders in your chosen design, and how does the selected library provide tools for addressing them (e.g., sensitivity analysis, instrumental variables, front-door adjustment)? Consider what methods are most appropriate based on the assumptions you are willing to make given the real-world scenario. How would you validate these assumptions?
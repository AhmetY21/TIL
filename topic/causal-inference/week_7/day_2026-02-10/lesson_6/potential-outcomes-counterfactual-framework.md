---
title: "Potential Outcomes (Counterfactual) Framework"
date: "2026-02-10"
week: 7
lesson: 6
slug: "potential-outcomes-counterfactual-framework"
---

# Topic: Potential Outcomes (Counterfactual) Framework

## 1) Formal definition (what is it, and how can we use it?)

The Potential Outcomes (also known as the Counterfactual) framework is a powerful tool for defining and estimating causal effects. It provides a rigorous way to think about what *would* have happened to a subject if they had received a different treatment than they actually did.

Here's the formal definition:

*   For each unit (e.g., person, experiment) *i* in a population, and for each possible treatment level *t* in a set of treatments *T*, there exists a *potential outcome*  Y<sub>i</sub>(t).  Y<sub>i</sub>(t) represents the outcome we *would* observe for unit *i* if it received treatment *t*.

*   The *Fundamental Problem of Causal Inference*: We can only observe one potential outcome for each unit. A unit either receives the treatment or doesn't.  We can never observe both Y<sub>i</sub>(t) and Y<sub>i</sub>(t') for t != t' for the same unit at the same time. This is why we need to make assumptions and use estimation strategies.

*   **Treatment effect:**  For a unit *i*, the individual treatment effect comparing treatment *t* to treatment *t'* is defined as Y<sub>i</sub>(t) - Y<sub>i</sub>(t').

*   **Average Treatment Effect (ATE):** The average treatment effect is the average of the individual treatment effects over the population: E[Y(t) - Y(t')].

*   **Average Treatment Effect on the Treated (ATT):** The average treatment effect on the treated is the average treatment effect for those who actually received the treatment *t*: E[Y(t) - Y(t') | T=t].

**How can we use it?**

The potential outcomes framework helps us:

*   **Clearly define causal effects:**  By focusing on what *would* have happened under different scenarios, it clarifies the causal question we are trying to answer.

*   **Identify assumptions needed for causal inference:**  Because we can't observe all potential outcomes, we need to make assumptions to estimate causal effects.  Common assumptions include:
    *   **Stable Unit Treatment Value Assumption (SUTVA):**  This assumption has two parts:
        *   No interference:  A unit's treatment does not affect the potential outcomes of other units.
        *   No multiple versions of treatment:  The treatment is the same for all units.
    *   **Ignorability (or Conditional Independence):** This assumption states that treatment assignment is independent of the potential outcomes, conditional on observed covariates.  Formally, (Y(0), Y(1)) ⊥ T | X, where T is the treatment indicator and X is a set of covariates.  This is often achieved through randomization in experiments or through careful covariate adjustment in observational studies.
    *   **Positivity (or Overlap):**  For every value of the covariates X, there is a non-zero probability of receiving each treatment level.  This ensures that there is overlap in the covariate distributions of the treated and control groups.

*   **Evaluate the plausibility of different causal identification strategies:** It provides a framework for evaluating whether different approaches (e.g., regression adjustment, propensity score matching, instrumental variables) can credibly estimate the desired causal effect given the data and assumptions.

## 2) Application scenario

**Scenario:**  We want to evaluate the effect of a new online advertising campaign on sales.

*   **Units:** Individual customers
*   **Treatment (T):** 1 = Customer saw the ad, 0 = Customer did not see the ad
*   **Outcome (Y):** Number of purchases made in a month

Using the Potential Outcomes framework:

*   Y<sub>i</sub>(1): The number of purchases customer *i* *would* have made if they *had* seen the ad.
*   Y<sub>i</sub>(0): The number of purchases customer *i* *would* have made if they *had not* seen the ad.

The individual treatment effect for customer *i* is Y<sub>i</sub>(1) - Y<sub>i</sub>(0).  The ATE is E[Y(1) - Y(0)], the average difference in purchases if everyone saw the ad versus if no one saw the ad.

**Challenges:**

We can only observe one of Y<sub>i</sub>(1) or Y<sub>i</sub>(0) for each customer.  We can't see what would have happened to a customer *if* they saw the ad if they actually *didn't*, and vice versa.

**Addressing the Challenges:**

*   **Randomized Experiment:**  If we randomly assign customers to see the ad or not, we can ensure (under SUTVA) that the ad assignment is independent of their potential outcomes. In this case, the observed difference in means between the treated and control groups is an unbiased estimate of the ATE.

*   **Observational Study:** If the ad exposure is not randomized, we need to account for confounding variables.  For example, customers who are more active online may be more likely to see the ad and more likely to make purchases regardless of the ad.  We would need to collect data on these confounders (e.g., website activity, demographics) and use methods like propensity score matching or regression adjustment to estimate the ATE, conditional on these observed covariates.  This relies on the ignorability assumption.

## 3) Python method (if possible)

While the Potential Outcomes framework is primarily conceptual, Python libraries like `CausalML`, `DoWhy`, `EconML`, and `Statsmodels` provide tools to estimate causal effects under this framework.  Here's an example using `DoWhy` to illustrate how propensity scores are used to adjust for confounders:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot

# Generate some synthetic data
np.random.seed(42)
n_samples = 1000
age = np.random.randint(20, 60, n_samples)
income = np.random.normal(50000 + 1000 * age, 10000, n_samples)
treatment_probability = 1 / (1 + np.exp(-(age - 40) / 5 + (income - 50000) / 20000))
treatment = np.random.binomial(1, treatment_probability, n_samples)
outcome = 0.1 * treatment + 0.001 * income + 0.05 * age + np.random.normal(0, 1, n_samples)

data = pd.DataFrame({'age': age, 'income': income, 'treatment': treatment, 'outcome': outcome})


# Create a causal model
model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['age', 'income']  # Specify confounders
)

# Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate the causal effect using propensity score weighting
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",  # Specify the method
    method_params={'weighting_func': "propensity"} # specify the weighting function
)

print(estimate)
```

**Explanation:**

1.  **Data Generation:**  We create a synthetic dataset where age and income are confounders affecting both the treatment and the outcome. The treatment variable is generated using a logistic function dependent on age and income, creating a relationship of influence.
2.  **Causal Model:** We create a `CausalModel` object, specifying the treatment, outcome, and common causes (confounders).
3.  **Identify Effect:** `model.identify_effect()` attempts to find a valid causal estimand, given the causal graph. The `proceed_when_unidentifiable=True` allows it to proceed even if full identification isn't possible.
4.  **Estimate Effect:** `model.estimate_effect()` estimates the causal effect using the specified method, which in this case is propensity score weighting.  Propensity score weighting aims to balance the observed covariates across treatment groups by weighting each unit by the inverse probability of receiving the treatment they actually received.

This example demonstrates how `DoWhy` leverages the potential outcomes framework by allowing you to explicitly define your causal model, identify the causal effect, and then estimate it using various methods that address confounding.

## 4) Follow-up question

How does the Potential Outcomes framework relate to the concept of causal graphs (e.g., Directed Acyclic Graphs or DAGs), and how can causal graphs help in identifying and estimating causal effects within the Potential Outcomes framework?
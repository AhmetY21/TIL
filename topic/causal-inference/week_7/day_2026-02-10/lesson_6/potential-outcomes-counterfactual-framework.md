---
title: "Potential Outcomes (Counterfactual) Framework"
date: "2026-02-10"
week: 7
lesson: 6
slug: "potential-outcomes-counterfactual-framework"
---

# Topic: Potential Outcomes (Counterfactual) Framework

## 1) Formal definition (what is it, and how can we use it?)

The Potential Outcomes (Counterfactual) Framework, also known as the Rubin Causal Model, is a statistical framework for defining and estimating causal effects. It revolves around the concept of "potential outcomes," which are the outcomes that *would* have been observed for each individual under different possible treatments or exposures.

**Formal Definition:**

For each individual *i*, let:

*   *T<sub>i</sub>* be the treatment assignment.  *T<sub>i</sub>* = 1 if the individual receives the treatment, and *T<sub>i</sub>* = 0 if the individual receives the control.
*   *Y<sub>i</sub>(1)* be the *potential outcome* for individual *i* if they were to receive the treatment (*T<sub>i</sub>* = 1). This is what *Y<sub>i</sub>* *would* be.
*   *Y<sub>i</sub>(0)* be the *potential outcome* for individual *i* if they were to receive the control (*T<sub>i</sub>* = 0). This is what *Y<sub>i</sub>* *would* be.
*   *Y<sub>i</sub>* be the observed outcome for individual *i*. We only observe one of the potential outcomes for each individual, depending on their treatment assignment:
    *   *Y<sub>i</sub>* = *Y<sub>i</sub>(1)* if *T<sub>i</sub>* = 1
    *   *Y<sub>i</sub>* = *Y<sub>i</sub>(0)* if *T<sub>i</sub>* = 0

The fundamental problem of causal inference is that we can only observe one potential outcome for each individual.  We can *never* observe *Y<sub>i</sub>(1)* and *Y<sub>i</sub>(0)* simultaneously for the same individual. This is the **fundamental problem of causal inference**.

**Causal Effect:**

The individual treatment effect (ITE) for individual *i* is defined as the difference between their potential outcomes under treatment and control:

*   ITE<sub>i</sub> = *Y<sub>i</sub>(1)* - *Y<sub>i</sub>(0)*

Since we can't observe both *Y<sub>i</sub>(1)* and *Y<sub>i</sub>(0)*, we can't directly compute the ITE for any individual. Therefore, we often focus on estimating the **Average Treatment Effect (ATE)**:

*   ATE = E[*Y<sub>i</sub>(1)*] - E[*Y<sub>i</sub>(0)*]

Where E[] denotes the expected value.

**How we use it:**

The potential outcomes framework allows us to formally define causal effects and think clearly about the assumptions required to estimate them. The framework relies on assumptions like:

*   **Stable Unit Treatment Value Assumption (SUTVA):** This has two components:
    *   **No Interference:** An individual's potential outcome is only affected by their own treatment assignment, not by the treatment assignments of others.
    *   **No Multiple Versions of Treatment:**  The treatment is consistently applied across individuals, and there are no hidden variations in the treatment that affect the outcome.
*   **Ignorability/Exchangeability:** Treatment assignment is independent of potential outcomes.  In other words, the treatment is assigned randomly, or we can control for confounders that affect both treatment assignment and potential outcomes.  Formally: *Y(0), Y(1)  ⊥ T | X*, where X are observed covariates.
*   **Positivity/Overlap:** For every value of the covariates *X*, there is a positive probability of receiving both treatment and control. This ensures we can compare individuals with similar characteristics under both treatment conditions.

By explicitly stating these assumptions, we can assess their plausibility in a given context and choose appropriate statistical methods to estimate causal effects.

## 2) Application scenario

Imagine we want to study the effect of a new drug on blood pressure.

*   *T<sub>i</sub>* = 1 if individual *i* receives the new drug, and *T<sub>i</sub>* = 0 if they receive a placebo.
*   *Y<sub>i</sub>(1)* is the blood pressure of individual *i* if they were to receive the new drug.
*   *Y<sub>i</sub>(0)* is the blood pressure of individual *i* if they were to receive the placebo.
*   *Y<sub>i</sub>* is the observed blood pressure of individual *i* after the treatment period.

We want to estimate the ATE, which is the average difference in blood pressure if everyone received the new drug versus if everyone received the placebo:  ATE = E[*Y<sub>i</sub>(1)*] - E[*Y<sub>i</sub>(0)*].

To do this, we can conduct a randomized controlled trial (RCT). If the treatment is randomly assigned, and SUTVA holds, we can estimate the ATE by simply taking the difference in the average blood pressure between the treatment and control groups:

Estimated ATE =  Mean blood pressure in the drug group - Mean blood pressure in the placebo group.

However, if treatment is NOT randomly assigned (e.g., doctors are more likely to prescribe the drug to patients with higher blood pressure), then we need to control for confounding variables (e.g., pre-existing health conditions, age, weight) using techniques like matching, propensity score weighting, or regression adjustment.

## 3) Python method (if possible)

While the potential outcomes framework itself is a conceptual framework, Python can be used to *estimate* causal effects within this framework, especially when dealing with observational data where treatment is not randomly assigned. One common approach is to use propensity score matching or weighting with libraries like `scikit-learn` and `statsmodels`.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Generate some simulated data (replace with your actual data)
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'age': np.random.randint(20, 70, n),
    'income': np.random.normal(50000, 20000, n),
    'health': np.random.normal(0, 1, n),
    'treatment': np.random.binomial(1, 0.3 + 0.2 * (np.random.rand(n) - 0.5), n) # Treatment is influenced by other factors
})
data['outcome'] = 2 * data['treatment'] + 0.5 * data['age'] - 0.3 * data['income'] + np.random.normal(0, 100, n)

# Propensity Score Estimation
X = data[['age', 'income', 'health']]
y = data['treatment']

# Using logistic regression to estimate propensity scores
model = LogisticRegression(random_state=42)
model.fit(X, y)
data['propensity_score'] = model.predict_proba(X)[:, 1]

# Inverse Probability of Treatment Weighting (IPTW)
data['iptw'] = np.where(data['treatment'] == 1, 1 / data['propensity_score'], 1 / (1 - data['propensity_score']))

# Estimate ATE using weighted regression
formula = 'outcome ~ treatment'
weighted_model = smf.wls(formula, data=data, weights=data['iptw']).fit()
print(weighted_model.summary())

# The coefficient for 'treatment' in the weighted model is the estimated ATE.
ate_estimate = weighted_model.params['treatment']
print(f"\nEstimated ATE: {ate_estimate}")


# Alternatively, using a simple difference in means (naive estimate - biased!)
ate_naive = data[data['treatment']==1]['outcome'].mean() - data[data['treatment']==0]['outcome'].mean()
print(f"Naive ATE estimate (biased): {ate_naive}")
```

This code demonstrates propensity score weighting, a common technique for causal inference under the potential outcomes framework.  It first estimates the propensity score (the probability of receiving treatment given observed covariates) using logistic regression. Then, it calculates inverse probability of treatment weights (IPTW) and uses them in a weighted regression model to estimate the ATE. The naive ATE calculation shows how simply comparing means without accounting for confounding can lead to biased results.  Many libraries exist for more robust matching and weighting techniques.

## 4) Follow-up question

How does the potential outcomes framework relate to other causal inference techniques like directed acyclic graphs (DAGs), and how can these different approaches be used together to strengthen causal claims? Specifically, how do DAGs help to identify confounders and mediators that need to be accounted for when estimating causal effects using potential outcomes?
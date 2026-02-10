---
title: "Potential Outcomes (Counterfactual) Framework"
date: "2026-02-10"
week: 7
lesson: 6
slug: "potential-outcomes-counterfactual-framework"
---

# Topic: Potential Outcomes (Counterfactual) Framework

## 1) Formal definition (what is it, and how can we use it?)

The potential outcomes framework, also known as the counterfactual framework or Rubin causal model, is a formal approach to defining and estimating causal effects.  It explicitly defines what it *means* to say that a treatment (or intervention) has a causal effect on an outcome.

The core idea is that for each individual unit (e.g., a person, a city, a product), there exist *potential outcomes* corresponding to different treatment states.  Specifically:

*   Let `i` index the unit.
*   Let `T` be the treatment variable. `T_i = 1` indicates that unit `i` received the treatment, and `T_i = 0` indicates that unit `i` did not receive the treatment (control group).
*   Let `Y` be the outcome variable.
*   `Y_i(1)` represents the potential outcome for unit `i` *if* they received the treatment.
*   `Y_i(0)` represents the potential outcome for unit `i` *if* they did not receive the treatment.

The *individual causal effect* for unit `i` is then defined as:

`τ_i = Y_i(1) - Y_i(0)`

In other words, the individual causal effect is the difference between what would have happened if the unit received the treatment and what would have happened if the unit did not receive the treatment.

The fundamental problem of causal inference is that we can only observe *one* of these potential outcomes for each unit. We either observe `Y_i(1)` if `T_i = 1`, or we observe `Y_i(0)` if `T_i = 0`. We can never observe both for the same unit. The other potential outcome is *counterfactual* because it describes what *would have* happened under a different treatment assignment.

We typically focus on estimating *average* causal effects. The most common is the Average Treatment Effect (ATE):

`ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]`

Where E[] denotes the expected value.  The ATE is the average difference in outcomes if everyone received the treatment versus if no one received the treatment.

**How we use it:**

1.  **Clearly Define Treatment & Outcome:**  Precisely specify the treatment (intervention) and the outcome variable.
2.  **Articulate Assumptions:** The potential outcomes framework forces you to make explicit assumptions about the relationship between treatment assignment and potential outcomes. The key assumption is often *ignorability* (or exchangeability), which states that treatment assignment is independent of the potential outcomes.  This means that treatment assignment is effectively random, conditional on observed covariates. Formally: `(Y(0), Y(1)) ⊥ T | X`, where `X` are observed covariates. Other assumptions include *stable unit treatment value assumption (SUTVA)*, which means no interference between units (one person's treatment doesn't affect another's outcome) and there are no multiple versions of treatment.
3.  **Estimate Average Treatment Effects:** Using statistical methods (e.g., regression, matching, inverse probability weighting) that adjust for confounders to estimate ATE or other treatment effects of interest (e.g., Average Treatment Effect on the Treated - ATT).

## 2) Application scenario

**Scenario:**  A company wants to evaluate the effectiveness of a new marketing campaign (the treatment) on increasing sales (the outcome).

*   **Units:** Individual customers.
*   **Treatment (T):** Whether or not a customer received the marketing campaign (`T_i = 1` if the customer received the campaign, `T_i = 0` otherwise).
*   **Outcome (Y):** The amount of money a customer spent on the company's products in a specific period.
*   **Y(1):** The amount the customer *would have* spent if they received the marketing campaign.
*   **Y(0):** The amount the customer *would have* spent if they did not receive the marketing campaign.

The company wants to estimate the ATE: `E[Y(1) - Y(0)]`. This represents the average increase in sales they would expect if they rolled out the marketing campaign to all their customers compared to if they did not roll it out at all.

To estimate this, they might use a randomized controlled trial (RCT). If the campaign assignment is truly random, then the treated and control groups should be similar on average, and the difference in average sales between the two groups will be an unbiased estimate of the ATE.

However, often assignment isn't random. Suppose the company targeted the campaign at customers who were already high spenders. In this case, a simple comparison of sales between those who received the campaign and those who did not would likely overestimate the effectiveness of the campaign.  The potential outcomes framework forces us to recognize that we need to adjust for these pre-existing differences (confounders) to get a valid estimate of the ATE. This could be done through regression adjustment, matching, or other causal inference methods.

## 3) Python method (if possible)

```python
import pandas as pd
import statsmodels.formula.api as smf

# Sample data (replace with your actual data)
data = {
    'customer_id': range(1, 101),
    'treatment': [0] * 50 + [1] * 50,  # 50 in control, 50 in treatment
    'sales': [10 + i for i in range(50)] + [15 + i + 5 for i in range(50)],  # sales for control, sales for treatment (with a boost)
    'past_spending': [5 + i*0.5 for i in range(50)] + [10 + i*0.5 for i in range(50)] # pre-treatment spending, higher for treatment group
}
df = pd.DataFrame(data)

# Simple difference in means (naive estimate - biased)
ate_naive = df[df['treatment'] == 1]['sales'].mean() - df[df['treatment'] == 0]['sales'].mean()
print(f"Naive ATE (difference in means): {ate_naive}")

# Regression adjustment to control for 'past_spending' (confounder)
# Assumes a linear relationship between past_spending, treatment, and sales
model = smf.ols("sales ~ treatment + past_spending", data=df)
results = model.fit()
print(results.summary())

# The coefficient for 'treatment' in the regression model is the adjusted ATE.
ate_adjusted = results.params['treatment']
print(f"Adjusted ATE (regression): {ate_adjusted}")

# Example using propensity score weighting (IPW).  First estimate propensity scores
from sklearn.linear_model import LogisticRegression
from statsmodels.api import add_constant
X = df[['past_spending']]
X = add_constant(X)
y = df['treatment']

propensity_model = LogisticRegression(random_state=0).fit(X, y)
propensity_scores = propensity_model.predict_proba(X)[:, 1] #probability of treatment

# Calculate IPW weights
df['ipw_weight'] = df['treatment']/propensity_scores + (1-df['treatment'])/(1-propensity_scores)

# Weighted ATE estimation
ate_ipw = (df[df['treatment'] == 1]['sales'] * df[df['treatment'] == 1]['ipw_weight']).mean() - (df[df['treatment'] == 0]['sales'] * df[df['treatment'] == 0]['ipw_weight']).mean()
print(f"Adjusted ATE (IPW): {ate_ipw}")
```

This code demonstrates a simple scenario and uses regression adjustment and IPW to estimate the ATE. It highlights how controlling for confounders (e.g., `past_spending`) can lead to a different (and hopefully less biased) estimate of the treatment effect.  Libraries like `CausalML`, `DoWhy`, and `EconML` offer more advanced methods and tools for causal inference in Python.

## 4) Follow-up question

How does the potential outcomes framework relate to other causal inference techniques, such as instrumental variables or front-door adjustment?  Specifically, how can we use the language of potential outcomes to understand the assumptions that underlie these other methods, and what are the advantages and disadvantages of each approach?
---
title: "Assumptions: SUTVA, Consistency, Exchangeability, Positivity"
date: "2026-02-13"
week: 7
lesson: 3
slug: "assumptions-sutva-consistency-exchangeability-positivity"
---

# Topic: Assumptions: SUTVA, Consistency, Exchangeability, Positivity

## 1) Formal definition (what is it, and how can we use it?)

Causal inference relies heavily on several key assumptions to allow us to draw valid conclusions about cause-and-effect relationships from observational or experimental data. These assumptions, while often implicit, are crucial for the validity of any causal analysis. The following are four common assumptions:

*   **SUTVA (Stable Unit Treatment Value Assumption):** This is actually *two* assumptions rolled into one:

    *   *No Interference:* A subject's treatment only affects their own outcome and not the outcome of any other subject.  In other words, the treatment assigned to one unit doesn't impact another unit's outcome.
    *   *Single Version of Treatment:*  There are no hidden variations of the treatment that affect the outcome differently.  The treatment is well-defined, and all units receiving the treatment experience the same "version" of it.

    *Use:* SUTVA allows us to define potential outcomes for each unit independent of the treatment assignment of other units.  If violated, causal estimates can be biased as observed effects might be due to spillover or varying treatment forms rather than the treatment itself.

*   **Consistency:** The potential outcome under the treatment a unit *actually* receives is equal to the observed outcome for that unit.  Formally, if *Z<sub>i</sub>* is the treatment received by unit *i*, and *Y<sub>i</sub>* is the observed outcome, then *Y<sub>i</sub> = Y<sub>i</sub>(Z<sub>i</sub>)*, where *Y<sub>i</sub>(Z<sub>i</sub>)* is the potential outcome for unit *i* if they received treatment *Z<sub>i</sub>*.

    *Use:* Consistency links potential outcomes to observed outcomes.  Without consistency, potential outcomes become purely hypothetical, making causal inference based on observed data impossible. It states that what *would* happen under the treatment *actually* received is what *did* happen.

*   **Exchangeability (also called Unconfoundedness, No Unmeasured Confounding, Ignorability):** Given a set of observed covariates *X*, the treatment assignment *Z* is independent of the potential outcomes *Y(0)* and *Y(1)*. In notation: *(Y(0), Y(1)) ⊥ Z | X*.  This means that after controlling for *X*, the treated and untreated groups are comparable.  All confounders (variables that affect both treatment and outcome) must be included in *X*.

    *Use:* Exchangeability allows us to estimate the causal effect of a treatment by comparing the outcomes of treated and untreated groups within strata defined by *X*.  If exchangeability doesn't hold (i.e., there are unmeasured confounders), then any observed differences in outcomes between the groups might be due to these unmeasured confounders rather than the treatment.

*   **Positivity (also called Overlap):** For every value of the covariates *X*, there is a non-zero probability of receiving each treatment level. Formally, *0 < P(Z = z | X = x) < 1* for all values *x* and *z*. This means there must be some individuals in each stratum of *X* who receive each treatment.

    *Use:* Positivity ensures that we can actually compare treated and untreated groups within each stratum of *X*. If positivity is violated (e.g., everyone with a certain value of *X* always receives the treatment), we cannot estimate the causal effect for that stratum.

## 2) Application scenario

Let's consider evaluating the causal effect of a new medication on blood pressure reduction.

*   **SUTVA:**  If patients discuss the medication and change their diets or exercise habits based on these discussions (interference), or if there are different batches of the medication with varying potencies (multiple versions of treatment), SUTVA is violated.

*   **Consistency:** If a patient took the medication as prescribed (Z=1), then their observed blood pressure reduction is what *would* happen if they took the medication (Y(1)). If they didn't take the medication as prescribed (e.g., only took it sporadically), then consistency becomes less certain.

*   **Exchangeability:**  If healthier individuals are more likely to be prescribed the medication (due to doctor's preferences or patient's health-seeking behavior), and we don't account for pre-existing health conditions (e.g., diabetes, heart disease) in our analysis (X), then exchangeability is violated.  Pre-existing health conditions are confounders, affecting both the likelihood of receiving the treatment and the outcome (blood pressure reduction).

*   **Positivity:**  If, based on some characteristic X, all individuals with very high blood pressure *always* get the new medication, and nobody with that characteristic is *ever* in the control group, then positivity is violated for that value of X. We can't estimate the medication's effect for that subpopulation.
## 3) Python method (if possible)

While there isn't a direct Python function to *check* these assumptions (they are generally untestable from data alone), Python libraries for causal inference provide tools that can help assess *sensitivity* to violations.

Specifically, for Exchangeability, libraries like `CausalML` or `EconML` help evaluate the impact of unobserved confounders. For positivity, libraries can help identify areas where propensity scores are too close to 0 or 1.

Here is an example using `statsmodels` to examine covariate overlap (related to positivity) after propensity score matching:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2)  # Covariates
treatment_effect = 0.5
propensity_score = 1 / (1 + np.exp(-(X[:, 0] - 0.5)))
treatment = np.random.binomial(1, propensity_score)
outcome = 2 * X[:, 0] + X[:, 1] + treatment_effect * treatment + np.random.normal(0, 0.5, n_samples)

data = pd.DataFrame(X, columns=['X1', 'X2'])
data['treatment'] = treatment
data['outcome'] = outcome

# Propensity score estimation using logistic regression
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(data[['X1', 'X2']], data['outcome'], data['treatment'], test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

propensity_model = LogisticRegression(random_state=42)
propensity_model.fit(X_train, t_train)
propensity_scores = propensity_model.predict_proba(X_test)[:, 1]

# Examine Propensity Score distribution to evaluate Positivity
print("Min propensity score:", np.min(propensity_scores))
print("Max propensity score:", np.max(propensity_scores))

# Check covariate balance AFTER matching (simplified example using inverse propensity weighting for demonstration)
# In practice, you'd perform actual matching or weighting
weights = data['treatment']/ propensity_score + (1-data['treatment']) / (1- propensity_score)

# Weighted means for treatment and control groups
weighted_mean_x1_treated = np.mean(data[data['treatment'] == 1]['X1'] * weights[data['treatment']==1])
weighted_mean_x1_control = np.mean(data[data['treatment'] == 0]['X1'] * weights[data['treatment']==0])

print(f"Weighted mean of X1 for treated: {weighted_mean_x1_treated}")
print(f"Weighted mean of X1 for control: {weighted_mean_x1_control}")

# Regression adjustment to estimate treatment effect (after addressing confounding)
model = sm.WLS(data['outcome'], sm.add_constant(data[['treatment', 'X1', 'X2']]), weights=weights)
results = model.fit()
print(results.summary())
```

This code demonstrates how to:

1.  Generate synthetic data with covariates, treatment, and outcome.
2.  Estimate propensity scores using logistic regression.
3.  Print the minimum and maximum propensity scores to assess positivity. Values very close to 0 or 1 indicate a violation.
4.  Calculates the means of the covariates for treatment and control groups using the weights from IPW.
5.  Estimate the causal effect by running a WLS regression.

## 4) Follow-up question

What are some strategies for addressing violations of each of these assumptions (SUTVA, Consistency, Exchangeability, Positivity) in a real-world causal inference problem?  For example, if I suspect interference is occurring, what could I do? If I suspect unmeasured confounding, what are my options?
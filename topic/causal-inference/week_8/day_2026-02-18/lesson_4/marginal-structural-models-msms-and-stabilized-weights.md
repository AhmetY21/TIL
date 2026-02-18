---
title: "Marginal Structural Models (MSMs) and Stabilized Weights"
date: "2026-02-18"
week: 8
lesson: 4
slug: "marginal-structural-models-msms-and-stabilized-weights"
---

# Topic: Marginal Structural Models (MSMs) and Stabilized Weights

## 1) Formal definition (what is it, and how can we use it?)

Marginal Structural Models (MSMs) are a class of causal inference models used to estimate the causal effect of a *time-varying* treatment on an outcome, in the presence of time-varying confounding.  Traditional regression models often fail in this scenario because they don't adequately account for the feedback loop where past treatment influences future confounders, which in turn influences future treatment decisions. This feedback loop can introduce bias.

MSMs address this by using inverse probability of treatment weighting (IPTW). The core idea is to create a *pseudo-population* where treatment assignment is independent of past confounding. This is achieved by weighting each individual's data by the inverse probability of the treatment they actually received, *conditional on their history of confounders and treatment*.  By weighting, we effectively simulate a randomized controlled trial where treatment is not influenced by these confounders.

**Formally:**

Let:
*   `A_t` be the treatment at time `t`.
*   `L_t` be the time-varying confounders at time `t`.
*   `Y` be the outcome.
*   `H_t` represent the history of treatment and confounders up to time t:  `H_t = (L_0, A_0, L_1, A_1, ..., L_t, A_t)`.

The causal effect of interest is typically defined using a potential outcome framework.  For example, if we are interested in the effect of a treatment regimen `a = (a_0, a_1, ..., a_T)` on the outcome `Y`, we want to estimate `E[Y^a]`, the expected outcome had everyone followed treatment regimen `a`.

MSMs estimate this by using a weighted regression of the outcome `Y` on the treatment history `a`:

`E[Y^a] ≈  Σ[ w * Y * I(A = a) ] / Σ[w * I(A = a)]`

Where:
* `w` are the weights.  The standard IPTW weights are calculated as the inverse probability of receiving the observed treatment, given the *past* history of treatment and confounders:

`w_t = Π_{k=0}^t P(A_k = a_k | H_{k-1})^-1` (where `H_{-1}` is empty or contains baseline covariates). These are the unstabilized weights.

**Stabilized Weights:**

Unstabilized weights can have large variances, leading to unstable effect estimates. Stabilized weights are used to reduce this variance. They adjust the unstabilized weights by multiplying by the *marginal* probability of treatment at each time point.

`w_t_stabilized = Π_{k=0}^t  P(A_k = a_k) / P(A_k = a_k | H_{k-1})`

where the numerator is the *marginal* probability of the observed treatment at each time, unconditional on any confounders.  This is usually estimated using a simpler model like a logistic regression of treatment at time t only on baseline covariates.

**How to use MSMs:**

1.  **Define the causal question:** Clearly specify the treatment and outcome of interest, and the potential treatment regimens to compare.
2.  **Identify time-varying confounders:** These are variables that are both influenced by past treatment and influence future treatment decisions and the outcome.
3.  **Estimate the conditional probability of treatment:**  Model `P(A_t | H_{t-1})` for each time point `t`. This often involves using logistic regression or other appropriate classification models.
4.  **Estimate the marginal probability of treatment (for stabilized weights):** Model `P(A_t)`. This is usually a simpler model than the conditional probability model, often involving just baseline covariates.
5.  **Calculate the weights:** Compute either the unstabilized or stabilized weights.
6.  **Fit the MSM:** Use a weighted regression model to estimate the effect of the treatment on the outcome, using the calculated weights. The treatment is a fixed regime, thus the indicator variable is needed to denote those who followed the fixed regime from the entire sample.
7.  **Interpret the results:** The coefficients from the weighted regression model can be interpreted as causal effects, under the assumption that all relevant confounders have been accounted for.

## 2) Application scenario

**Scenario:** Suppose we want to study the causal effect of antiretroviral therapy (ART) initiation timing on mortality among HIV-positive individuals. ART initiation is a time-varying treatment because individuals can initiate ART at different points in time. CD4 count is a time-varying confounder because it's affected by previous ART treatment and it influences future treatment decisions (doctors are more likely to prescribe ART to patients with lower CD4 counts) and affects mortality. Viral load is another time-varying confounder with a similar effect.

**Why MSMs are needed:** Simple regression models would likely be biased because they don't account for the complex interplay between ART initiation, CD4 count, viral load, and mortality. Individuals who initiate ART later may have lower CD4 counts and higher viral loads, making them more likely to die even if ART is effective.  By using MSMs and IPTW, we can adjust for this confounding and estimate the causal effect of different ART initiation strategies on mortality.

We can compare, for example, the regime of starting ART immediately versus delaying ART until a certain CD4 count threshold. We would estimate the weights using models for the probability of ART initiation at each time, conditional on past CD4 count and viral load. The MSM would then be a weighted regression of mortality on the chosen ART initiation regime.

## 3) Python method (if possible)

While a single, comprehensive MSM package doesn't exist in base Python, several libraries facilitate the necessary steps.  `statsmodels` and `scikit-learn` can be used to build the models needed for weight calculation, and `statsmodels` can then be used for the weighted regression. `Lifelines` is helpful if mortality is the outcome and needs to be modeled as a time to event.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

# Sample data (replace with your actual data)
# Each row represents a person at a time point
data = pd.DataFrame({
    'person_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
    'art': [0, 0, 1, 0, 1, 1, 0, 0, 0], # 1 = on ART, 0 = not on ART
    'cd4': [500, 400, 300, 600, 500, 400, 700, 600, 500],
    'viral_load': [10000, 20000, 5000, 5000, 1000, 200, 100, 50, 10],
    'mortality': [0, 0, 1, 0, 0, 0, 0, 0, 0]  # 1 = died, 0 = survived
})

# Lag CD4 and viral load to avoid future confounding
data['cd4_lagged'] = data.groupby('person_id')['cd4'].shift(1)
data['viral_load_lagged'] = data.groupby('person_id')['viral_load'].shift(1)
data = data.dropna()  # Remove first time point for each person


# 1. Estimate conditional probabilities of treatment (P(A_t | H_{t-1}))
#   - Logistic regression model for each time point
def fit_treatment_model(df, time):
    df_time = df[df['time'] == time]
    X = df_time[['cd4_lagged', 'viral_load_lagged']]  # Confounders (history)
    y = df_time['art']
    model = LogisticRegression(solver='liblinear').fit(X, y)  # Logistic regression
    return model

treatment_models = {}
for t in data['time'].unique():
    treatment_models[t] = fit_treatment_model(data, t)

# Function to predict treatment probability
def predict_treatment_prob(model, cd4, viral_load):
    # Create a DataFrame mimicking the data format
    X = pd.DataFrame({'cd4_lagged': [cd4], 'viral_load_lagged': [viral_load]})
    return model.predict_proba(X)[:, 1][0]  # Probability of treatment = 1


# Calculate propensity scores using the trained models
data['propensity_score'] = data.apply(lambda row: predict_treatment_prob(
    treatment_models[row['time']], row['cd4_lagged'], row['viral_load_lagged']
) if row['art'] == 1 else (1 - predict_treatment_prob(
    treatment_models[row['time']], row['cd4_lagged'], row['viral_load_lagged']
)), axis=1)

# Ensure no extreme propensity scores
data['propensity_score'] = np.clip(data['propensity_score'], 0.01, 0.99)

# Calculate inverse probability of treatment weights (unstabilized)
data['iptw'] = 1 / data['propensity_score']


# 2. Estimate marginal probabilities of treatment (P(A_t)) - for stabilized weights
# Logistic regression using only baseline covariates - let's use CD4 at baseline here for simplicity
# In a real analysis, you'd likely only use TIME-INVARIANT baseline covariates here!
baseline_cd4 = data.groupby('person_id')['cd4'].first().rename('baseline_cd4')
data = data.merge(baseline_cd4, left_on='person_id', right_index=True)

marginal_treatment_models = {}
for t in data['time'].unique():
    df_time = data[data['time'] == t]
    X = df_time[['baseline_cd4']]  # Baseline confounders
    y = df_time['art']
    model = LogisticRegression(solver='liblinear').fit(X, y)  # Logistic regression
    marginal_treatment_models[t] = model

def predict_marginal_treatment_prob(model, baseline_cd4):
    X = pd.DataFrame({'baseline_cd4': [baseline_cd4]})
    return model.predict_proba(X)[:, 1][0]

data['marginal_propensity_score'] = data.apply(lambda row: predict_marginal_treatment_prob(
    marginal_treatment_models[row['time']], row['baseline_cd4']
) if row['art'] == 1 else (1 - predict_marginal_treatment_prob(
    marginal_treatment_models[row['time']], row['baseline_cd4']
)), axis=1)

#Ensure no extreme marginal propensity scores
data['marginal_propensity_score'] = np.clip(data['marginal_propensity_score'], 0.01, 0.99)

# Calculate stabilized IPTW
data['iptw_stabilized'] = data['marginal_propensity_score'] / data['propensity_score']


# 3. Fit the MSM (weighted regression)
#   - Example: Effect of ART on mortality

#Define a treatment regime - start at T=1.
data['regime'] = data.apply(lambda row: 1 if row['time'] >=1 else 0, axis=1)

# Indicator variable for those who followed regime
data['regime_follower'] = data['regime'] == data['art']


#Unstabilized MSM
model_unstabilized = smf.glm("mortality ~ regime_follower", data=data, weights=data['iptw'], family=sm.families.Binomial()).fit()
print("Unstabilized MSM results:")
print(model_unstabilized.summary())


# Stabilized MSM
model_stabilized = smf.glm("mortality ~ regime_follower", data=data, weights=data['iptw_stabilized'], family=sm.families.Binomial()).fit()
print("\nStabilized MSM results:")
print(model_stabilized.summary())


# Interpretation:
#   - The coefficient for 'regime_follower' estimates the causal effect of the treatment
#     regime (starting ART at T=1).  If it's negative, it suggests that following the
#     regime (starting ART at T=1) is associated with lower mortality.
```

**Explanation:**

*   **Data Preparation:** The code simulates data representing individuals followed over time, including treatment status (`art`), time-varying confounders (`cd4`, `viral_load`), and the outcome (`mortality`).
*   **Conditional Probability Estimation:**  Logistic regression is used to model the probability of receiving ART at each time point, given past CD4 count and viral load.
*   **Marginal Probability Estimation:**  Logistic regression is used to model the probability of receiving ART at each time point, given *only* baseline CD4 count.
*   **Weight Calculation:** The inverse probability of treatment weights (IPTW) are calculated, both unstabilized and stabilized.
*   **MSM Fitting:** A weighted logistic regression model is fitted to estimate the effect of a ART initiation regime (starting at time 1) on mortality, using the calculated IPTW weights.  The `glm` function from `statsmodels` is used for weighted regression.
*   **Interpretation:**  The coefficient of the treatment variable in the weighted regression represents the estimated causal effect.

**Important Notes:**

*   This is a simplified example. Real-world applications require careful consideration of the specific causal question, identification of relevant confounders, and appropriate model specification.
*   The example uses logistic regression for weight estimation. Other models, such as decision trees or random forests, could be used depending on the nature of the data and the complexity of the relationships between variables.
*   The MSM is a regression model, it can use other families than the Binomial family. In these cases, you'd need to adjust the response variable accordingly.
*   It is crucial to check for positivity violations (individuals with extreme weights). Truncating or trimming weights may be necessary.
* The marginal probability estimation here is done with only baseline covariates, but that doesn't always have to be the case.

## 4) Follow-up question

How do I assess the positivity assumption (also known as the overlap assumption) when using MSMs, and what strategies can I use if positivity is violated?  Provide concrete examples of violations that might arise in the ART initiation scenario discussed above.
---
title: "Measurement Error, Missing Data, and Selection Bias (Causal View)"
date: "2026-02-19"
week: 8
lesson: 1
slug: "measurement-error-missing-data-and-selection-bias-causal-view"
---

# Topic: Measurement Error, Missing Data, and Selection Bias (Causal View)

## 1) Formal definition (what is it, and how can we use it?)

In causal inference, understanding and addressing measurement error, missing data, and selection bias is crucial for obtaining accurate estimates of causal effects. The causal view emphasizes that these issues should be analyzed within a causal framework, often leveraging causal diagrams (Directed Acyclic Graphs - DAGs) to represent relationships between variables.

**Measurement Error:** This occurs when the observed value of a variable differs from its true value. It can be random or systematic (biased). In a causal framework, we consider how measurement error affects the identification and estimation of causal effects.  The key idea is that measurement error acts as a type of "noise" that can attenuate or distort the estimated effect.  If the error is correlated with other variables in the DAG (e.g., the true value of the measured variable, the treatment, or the outcome), it can create spurious associations and bias causal estimates. Representing measurement error in the DAG is important; we would typically have a node representing the *true* variable and another node representing the *measured* variable, with an arrow from the true variable to the measured variable.

**Missing Data:** Data is missing when we do not observe values for certain variables for some individuals. The crucial distinction lies in *why* the data is missing. Rubin's classification distinguishes between:

*   **Missing Completely at Random (MCAR):** The probability of missingness is unrelated to any observed or unobserved variables. This is the simplest case.
*   **Missing at Random (MAR):** The probability of missingness depends only on observed variables. We can condition on these observed variables to correct for the missingness.
*   **Missing Not at Random (MNAR):** The probability of missingness depends on the unobserved value of the variable that is missing *itself*. This is the most challenging case because we cannot directly observe the factors influencing missingness.  Causal diagrams help visualize if missingness induces dependence between variables that would otherwise be independent.

Representing missingness in a DAG can be achieved by introducing a missingness indicator variable (e.g.,  `R_Y = 1` if Y is missing, 0 otherwise). The arrows into `R_Y` then indicate the variables influencing the missingness.

**Selection Bias:** This occurs when the sample we analyze is not representative of the population to which we want to generalize our causal findings. Selection bias arises when the process of selecting individuals into the sample is related to both the treatment and the outcome.  This induces a spurious association between the treatment and outcome in the observed sample. Selection bias is represented in a DAG by a collider node (often a selection indicator, `S=1` if selected, `S=0` otherwise) that has arrows pointing into it from both the treatment and the outcome (or variables that influence them). Conditioning on this collider induces a relationship between the treatment and the outcome even if they are causally independent in the overall population.

**How can we use it?**
By representing these issues within a causal framework (using DAGs), we can:

*   Identify potential sources of bias.
*   Determine what variables need to be adjusted for (e.g., via covariate adjustment, inverse probability weighting, or instrumental variables) to obtain unbiased causal estimates.
*   Assess the plausibility of different assumptions about the missing data mechanism (MAR vs. MNAR).
*   Conduct sensitivity analyses to assess how robust our conclusions are to different assumptions about the magnitude and nature of measurement error or the MNAR mechanism.

## 2) Application scenario

**Scenario: Effect of a Job Training Program on Earnings, with Measurement Error, Missing Data, and Selection Bias**

Let's say we are evaluating the impact of a job training program (Treatment, T) on subsequent earnings (Outcome, Y).

*   **Measurement Error:** Earnings (Y) are self-reported and may be subject to measurement error due to recall bias or strategic misreporting.  Let's say there's an underlying "true" earning Y*, but we observe Y = Y* + error. This error could be correlated with pre-training income (Z).

*   **Missing Data:** Some individuals drop out of the study before earnings are measured (missing Y). Suppose individuals with lower motivation (U - an unobserved confounder) are more likely to drop out and also less likely to have high earnings, *even if* they completed the training.  This is an example of MNAR since the missingness is related to the unobserved value of Y *and* a confounder.

*   **Selection Bias:** Participation in the job training program (T) is not random. Individuals are *selected* into the program (S=1) based on their skills and unemployment status (Z, observed).  Skills and unemployment status also affect their potential earnings. Therefore, conditioning on being in the *selected* sample (S=1) induces a relationship between T and Y.

In this scenario, we need to consider the following:

1.  **Measurement Error:**  We might try to use an instrumental variable for true earnings (Y*) if one is available.  Alternatively, sensitivity analysis can be used to assess how different levels of measurement error might bias the results.

2.  **Missing Data:**  We might attempt to use inverse probability weighting (IPW) to correct for missing data if we believe the data are MAR. This requires modeling the probability of observing earnings as a function of observed covariates (Z, T) and weighting the observed data by the inverse of that probability.  If we suspect MNAR, we need more sophisticated methods or sensitivity analysis under different MNAR assumptions.

3.  **Selection Bias:**  We need to carefully control for variables that influence both selection into the program and earnings (Z).  This could involve covariate adjustment, propensity score methods, or other techniques to address confounding.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils import resample
import statsmodels.api as sm

# Simulate data
np.random.seed(42)
n = 1000
Z = np.random.normal(0, 1, n)  # Observed confounder (skills, unemployment status)
U = np.random.normal(0, 1, n)  # Unobserved confounder (motivation)
T_prob = 1 / (1 + np.exp(-Z + 0.5*U)) # Probability of treatment assignment
T = np.random.binomial(1, T_prob, n)  # Treatment (job training)
Y_star = 2 + 1*T + 0.5*Z + 0.8*U + np.random.normal(0, 1, n)  # True earnings
error = np.random.normal(0, 0.5, n) # Measurement error
Y = Y_star + error # Measured earnings

# Simulate missing data (MNAR)
missing_prob = 1 / (1 + np.exp(-0.5*U + 0.2*Y_star)) # Probability of missingness depends on U and Y_star
R_Y = np.random.binomial(1, missing_prob, n) # Missingness indicator
Y[R_Y == 1] = np.nan

# Simulate selection bias
select_prob = 1 / (1 + np.exp(-1 + T + 0.5*Z)) # Probability of being selected based on T and Z
S = np.random.binomial(1, select_prob, n) # Selection indicator
T_obs = T[S == 1] # Treatement for the selected population
Y_obs = Y[S == 1] # Earnings for the selected population
Z_obs = Z[S == 1] # Skills for the selected population

# Create DataFrame
df = pd.DataFrame({'T': T, 'Y': Y, 'Z': Z, 'U': U, 'Y_star': Y_star, 'R_Y': R_Y, 'S':S})
df_obs = pd.DataFrame({'T': T_obs, 'Y': Y_obs, 'Z': Z_obs})

# Naive estimate (ignoring measurement error, missing data, and selection bias)
model_naive = LinearRegression()
model_naive.fit(df_obs[['T']], df_obs['Y'].fillna(df_obs['Y'].mean()))
print("Naive estimate (selected population, missing data imputed with mean):", model_naive.coef_[0])

# Estimate with IPW for missing data
df_complete = df.dropna(subset=['Y'])
X = df_complete[['T', 'Z']]
y = df_complete['R_Y']

# Fit logistic regression model for propensity to not be missing
missing_model = LogisticRegression()
missing_model.fit(X, 1-y)

# Predicted probabilities
propensity_scores = missing_model.predict_proba(df[['T', 'Z']])[:, 1]
df['ipw'] = 1 / propensity_scores

# Estimate using IPW
model_ipw = sm.WLS(df['Y'], sm.add_constant(df['T']), weights=df['ipw'], missing='drop').fit()
print("IPW estimate (missing data addressed, but selected population):", model_ipw.params['T'])

# Estimate using covariate adjustment to partially address selection bias in the IPW population
model_ca = sm.OLS(df_complete['Y'], sm.add_constant(df_complete[['T', 'Z']]), missing='drop').fit()
print("Covariate Adjustment Estimate with IPW in the non-missing population:", model_ca.params['T'])

# Note that we would need to either simulate from the true underlying distribution
# after addressing selection and/or use an instrument to remove the effect
# of measurement error to obtain a more accurate effect estimate.
```

**Explanation:**

*   **Data Simulation:**  The code simulates a dataset with treatment, outcome, confounders (observed and unobserved), measurement error, missing data, and selection.  The relationships between variables are defined according to the scenario described above.

*   **Naive Estimate:**  This simply regresses the (observed, incomplete) earnings on the treatment, ignoring all the potential biases.  The missing data are filled with the mean, which can introduce bias. It's also done on the selected population.

*   **IPW:** Inverse Probability Weighting is used to address the missing data, assuming MAR (conditional on T and Z).  A logistic regression model is fit to predict the probability of *not* being missing, and weights are calculated as the inverse of these probabilities.  A weighted least squares regression is then used to estimate the treatment effect.  However, the analysis is still performed on the *selected* population.

*   **Covariate Adjustment:** We implement covariate adjustment for Z in the IPW weighted regression. This can reduce bias associated with the selection.

*   **Limitations:** This is a simplified illustration. Addressing MNAR and measurement error fully requires more sophisticated techniques or stronger assumptions.

## 4) Follow-up question

Given the Python code provided, how could sensitivity analysis be used to assess the potential impact of *unobserved* confounding (U) on the estimated causal effect of the job training program, especially when the missingness mechanism is suspected to be MNAR, given we have a model for IPW? Specifically, outline the steps required to conduct a sensitivity analysis to assess how different correlations between U and the missingness indicator R_Y, and U and the outcome Y_star, might affect the IPW adjusted estimate.
---
title: "Propensity Score Matching (PSM)"
date: "2026-02-12"
week: 7
lesson: 4
slug: "propensity-score-matching-psm"
---

# Topic: Propensity Score Matching (PSM)

## 1) Formal definition (what is it, and how can we use it?)

Propensity Score Matching (PSM) is a statistical matching technique that attempts to estimate the effect of a treatment, intervention, or exposure by accounting for the covariates that predict receiving the treatment.  In essence, it aims to create a pseudo-randomized control trial from observational data, mitigating selection bias and confounding.

The *propensity score* is the conditional probability of receiving the treatment given a set of observed covariates. Mathematically, it is defined as:

`e(X) = P(T = 1 | X)`

Where:
*   `e(X)` is the propensity score.
*   `T` is the treatment indicator (1 = treated, 0 = control).
*   `X` is the vector of observed covariates.

PSM works by matching treated individuals with control individuals who have similar propensity scores. After matching, we can compare the outcomes of the matched treated and control groups to estimate the average treatment effect on the treated (ATT).  There are several matching algorithms used in PSM:

*   **Nearest Neighbor Matching:**  Each treated unit is matched to the closest control unit based on propensity score. Variants include matching *with replacement* (a control unit can be matched to multiple treated units) and *without replacement* (a control unit can only be matched once).

*   **Calipers:** A maximum allowable difference (caliper) in propensity scores is imposed for a match to be considered valid. This prevents matches that are too dissimilar.

*   **Radius Matching:**  Each treated unit is matched to all control units within a specific radius of its propensity score.

*   **Kernel Matching:** Uses a weighted average of the outcomes of all control units, with weights proportional to the similarity of their propensity scores to the treated unit.

*   **Stratification (Subclassification):**  The data is divided into strata based on propensity score ranges, and the treatment effect is estimated separately within each stratum.

The key assumption underlying PSM is *conditional independence* (also known as unconfoundedness or ignorability):  Given the observed covariates, treatment assignment is independent of the potential outcomes.  In other words, all confounding variables have been observed and included in the propensity score model. If there are unobserved confounders, PSM will not produce unbiased estimates.

We can use PSM to:

*   Estimate the causal effect of a program or policy.
*   Reduce bias in observational studies.
*   Compare the outcomes of different groups when random assignment is not possible.

## 2) Application scenario

Imagine we want to study the impact of a new job training program on individuals' income. We cannot randomly assign people to participate in the program. Instead, we observe who chooses to participate (the treated group) and compare their income to those who did not participate (the control group).

People who choose to participate in the training program might be systematically different from those who don't. For example, they might be more motivated, have lower initial skills, or live in areas with limited job opportunities. These differences could confound the relationship between the training program and income, leading to biased estimates.

Using PSM, we would:

1.  Collect data on relevant covariates, such as education level, age, prior work experience, geographic location, and motivation.
2.  Estimate the propensity score (the probability of participating in the training program) based on these covariates using a logistic regression model.
3.  Match participants in the training program (treated group) to non-participants (control group) with similar propensity scores. For example, we might use nearest neighbor matching with a caliper to ensure the matches are close enough.
4.  Compare the income of the matched treated and control groups. The difference in income would be our estimate of the effect of the job training program on income.
5. Assess the balance of covariates between the treated and matched controls. Ideally, these groups should be more similar after matching. Common metrics include standardized mean differences and variance ratios.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# Sample data (replace with your actual data)
data = {
    'treatment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 = treated, 0 = control
    'age': [30, 25, 35, 28, 40, 32, 27, 38, 33, 29],
    'education': [12, 16, 14, 12, 18, 14, 10, 16, 12, 14],  # Years of education
    'income_pre': [30000, 40000, 35000, 32000, 50000, 38000, 25000, 45000, 33000, 36000],
    'outcome': [35000, 42000, 40000, 34000, 55000, 40000, 28000, 48000, 38000, 37000] # Income after
}
df = pd.DataFrame(data)

# 1. Estimate the propensity scores using logistic regression
X = df[['age', 'education', 'income_pre']]  # Covariates
y = df['treatment']  # Treatment indicator

model = LogisticRegression()
model.fit(X, y)
propensity_scores = model.predict_proba(X)[:, 1]  # Probabilities of being treated
df['propensity_score'] = propensity_scores

# Check the model's ability to discriminate between treated and control
auc = roc_auc_score(y, propensity_scores)
print(f"AUC: {auc}")

# 2. Perform nearest neighbor matching with a caliper
def perform_psm(df, caliper=0.2):
    """Performs propensity score matching using nearest neighbors."""
    treated_indices = df[df['treatment'] == 1].index
    control_indices = df[df['treatment'] == 0].index

    treated_ps = df.loc[treated_indices, 'propensity_score'].values.reshape(-1, 1)
    control_ps = df.loc[control_indices, 'propensity_score'].values.reshape(-1, 1)

    # Nearest Neighbors matching
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_ps)

    distances, indices = nn.kneighbors(treated_ps)

    # Apply caliper
    caliper_condition = distances <= caliper
    matched_indices = [control_indices[indices[i][0]] if caliper_condition[i][0] else None for i in range(len(treated_indices))]

    matched_df = df.loc[treated_indices, :].copy()
    matched_df['matched_control'] = matched_indices  # index of the matched control
    matched_df = matched_df[matched_df['matched_control'].notna()] # Drop treated individuals without matching
    matched_df['matched_control'] = matched_df['matched_control'].astype(int)

    return matched_df

matched_df = perform_psm(df)

# 3. Calculate the average treatment effect on the treated (ATT)
matched_df['outcome_control'] = matched_df['matched_control'].apply(lambda x: df.loc[x, 'outcome'])
matched_df['treatment_effect'] = matched_df['outcome'] - matched_df['outcome_control']
att = matched_df['treatment_effect'].mean()

print(f"Average Treatment Effect on the Treated (ATT): {att}")

# 4. Check balance (example: standardized mean difference for age)
def standardized_mean_difference(group1, group2):
    """Calculates the standardized mean difference."""
    mean1 = group1.mean()
    mean2 = group2.mean()
    std1 = group1.std()
    std2 = group2.std()
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std

# Before matching
before_match_smd_age = standardized_mean_difference(df[df['treatment'] == 1]['age'], df[df['treatment'] == 0]['age'])
print(f"SMD (Age) Before Matching: {before_match_smd_age}")

# After matching
treated_matched_age = matched_df['age']
control_matched_age = matched_df['matched_control'].apply(lambda x: df.loc[x, 'age'])
after_match_smd_age = standardized_mean_difference(treated_matched_age, control_matched_age)
print(f"SMD (Age) After Matching: {after_match_smd_age}")
```

**Explanation:**

1.  **Data Preparation:** Creates a sample DataFrame (replace this with your actual data).  Includes a `treatment` column (1 for treated, 0 for control), covariates (`age`, `education`, `income_pre`), and an `outcome` column (income after).
2.  **Propensity Score Estimation:**  Uses `LogisticRegression` from scikit-learn to estimate the propensity scores.  The covariates are used to predict the probability of being treated.  The predicted probabilities are stored in a new column called `propensity_score`.  Also calculates and prints the Area Under the ROC Curve (AUC) to assess the propensity score model's performance.  A good model should be able to distinguish between treated and control groups.
3.  **Nearest Neighbor Matching:** The `perform_psm` function does the matching using `NearestNeighbors`.  It finds the nearest control unit for each treated unit based on the propensity score, applying a `caliper` to restrict matching to reasonably similar units. The `caliper` is set to 0.2, which is a reasonable starting point.
4.  **ATT Calculation:** Calculates the Average Treatment Effect on the Treated (ATT) by comparing the outcomes of the matched treated and control groups.
5.  **Balance Checking:**  Includes an example of checking balance using the standardized mean difference (SMD). The goal of matching is to reduce the SMD between the treated and control groups for all covariates. You should check the balance for *all* relevant covariates after matching.

**Important Notes:**

*   This is a simplified example. You'll likely need to adapt the code to your specific dataset and research question.
*   Parameter Tuning:  The caliper value can significantly affect the results. Experiment with different values to find the one that provides the best balance without sacrificing too much sample size.
*   Balance Checking:  Rigorously check the balance of covariates *after* matching.  Use appropriate balance diagnostics like standardized mean differences, variance ratios, and visual inspections (e.g., histograms, boxplots).  If balance is poor, you may need to adjust the propensity score model (add interaction terms or higher-order terms), change the matching algorithm, or try different caliper values.
*   Sensitivity Analysis:  Since PSM relies on the assumption of no unobserved confounders, it's crucial to perform sensitivity analyses to assess how robust your results are to potential hidden biases. Methods such as Rosenbaum sensitivity analysis can be employed for this purpose.
*   Package Alternatives: Consider using more specialized causal inference packages like `causalml` or `EconML` for more advanced PSM implementations and balance diagnostics.
*   Consider using cross-validation to select the optimal caliper size to improve the robustness of the matching process.

## 4) Follow-up question

How do I assess the quality of the matching in PSM to determine if it effectively reduced bias?  What metrics and visualizations are most helpful, and what are the acceptable ranges for these metrics to ensure reliable causal inference?
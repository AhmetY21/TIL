---
title: "Propensity Score Matching (PSM)"
date: "2026-02-12"
week: 7
lesson: 3
slug: "propensity-score-matching-psm"
---

# Topic: Propensity Score Matching (PSM)

## 1) Formal definition (what is it, and how can we use it?)

Propensity Score Matching (PSM) is a statistical technique used in causal inference to estimate the effect of a treatment or intervention on an outcome when random assignment is not possible (observational studies).  It attempts to mimic a randomized controlled trial (RCT) by creating treatment and control groups that are balanced in terms of observed covariates.

**Formal Definition:** The propensity score, denoted as *e(X)*, is the conditional probability of receiving the treatment given a set of observed covariates *X*:

*e(X) = P(T = 1 | X)*

where:

*   *T* = 1 indicates treatment, and *T* = 0 indicates control.
*   *X* represents the set of pre-treatment covariates thought to influence both the treatment assignment and the outcome.

**How PSM Works:**

1.  **Propensity Score Estimation:** A statistical model (typically logistic regression) is used to estimate the propensity score for each individual based on their observed covariates.  This yields a predicted probability of treatment for each subject, even those in the control group.

2.  **Matching:** Individuals in the treatment group are then matched to individuals in the control group based on the similarity of their propensity scores.  Several matching algorithms exist, including:
    *   **Nearest Neighbor Matching:** Each treated individual is matched to the control individual with the closest propensity score. Variations exist with replacement (allowing controls to be matched to multiple treated individuals) and without replacement.
    *   **Optimal Matching:** Attempts to minimize the total distance (in propensity score) across all matched pairs. More computationally intensive.
    *   **Caliper Matching:** Matches individuals whose propensity scores are within a certain threshold (the caliper) of each other.  This helps ensure good matching quality.
    *   **Kernel Matching:** Uses all control units, weighted by the proximity of their propensity scores to the treated unit's propensity score.

3.  **Outcome Comparison:** After matching, the outcome variable is compared between the matched treatment and control groups.  The difference in outcomes is then attributed to the treatment effect, assuming that all relevant confounders were included in the covariates used to estimate the propensity score.

4.  **Balance Check:** It's crucial to assess whether the matching process has successfully balanced the covariates between the treatment and control groups.  This can be done by comparing the means and variances of the covariates in the matched samples.  If balance is not achieved, it suggests either that the covariates need to be adjusted, that the model used to estimate the propensity scores needs to be improved, or that PSM is simply not appropriate for the data.

**Assumptions of PSM:**

*   **Conditional Independence Assumption (CIA) / Unconfoundedness:** All relevant confounders (variables that affect both treatment assignment and the outcome) have been observed and included in the covariates *X*.  This is the most crucial and often the most difficult to verify.  If there are unobserved confounders, PSM will not produce unbiased estimates.
*   **Overlap (Common Support):** There is sufficient overlap in the covariate distributions between the treatment and control groups.  In other words, for every value of *X*, there should be a non-zero probability of receiving both treatment and control.  PSM cannot extrapolate treatment effects to regions of the covariate space where there are no comparable control individuals.
*   **Stable Unit Treatment Value Assumption (SUTVA):** Each individual's outcome depends only on their own treatment status, and not on the treatment status of others (no interference) and there are no different forms of treatment leading to different outcomes.

## 2) Application scenario

Consider a scenario where we want to evaluate the effectiveness of a job training program on increasing individual income. We have data on individuals who participated in the program (treatment group) and individuals who did not (control group).  However, participation in the job training program was not random. Individuals who enrolled might be more motivated, possess higher levels of education, or have a stronger desire to improve their employment prospects. These factors could also independently influence their income, creating confounding bias.

In this situation, we could use PSM to match individuals in the treatment group to similar individuals in the control group based on observed covariates such as education level, age, previous work experience, and other demographic characteristics.  By comparing the income of the matched treatment and control groups, we can obtain a more reliable estimate of the program's effect on income than simply comparing the overall average income of the two groups. We're essentially controlling for the selection bias by finding comparable control units for each treated unit.

## 3) Python method (if possible)

There are several Python libraries you can use for PSM, most notably `scikit-learn` and `statsmodels`, often used in conjunction. For more specialized functionality consider `causalinference`. Here's a basic example using `scikit-learn` for propensity score estimation and nearest neighbor matching:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report

# Generate some synthetic data
np.random.seed(42)
n_samples = 500
data = pd.DataFrame({
    'age': np.random.randint(20, 60, n_samples),
    'education': np.random.randint(8, 16, n_samples),
    'income_prior': np.random.randint(20000, 60000, n_samples),
})

# Simulate treatment assignment (influenced by covariates)
propensity = 0.2 + 0.01 * data['age'] - 0.005 * data['education'] + 0.00001 * data['income_prior']
treatment = np.random.binomial(1, propensity)
data['treatment'] = treatment

# Simulate outcome (influenced by treatment and covariates)
outcome = 5000 + 100 * data['age'] + 500 * data['education'] + 0.8 * data['income_prior'] + 3000 * data['treatment'] + np.random.normal(0, 5000, n_samples)
data['outcome'] = outcome

# Separate features and treatment
X = data[['age', 'education', 'income_prior']]
y = data['treatment']

# 1. Estimate Propensity Scores
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict propensity scores on the entire dataset
propensity_scores = logistic_model.predict_proba(X)[:, 1]
data['propensity_score'] = propensity_scores

# 2. Matching (Nearest Neighbor Matching)
treated_indices = data[data['treatment'] == 1].index
control_indices = data[data['treatment'] == 0].index

# Create NearestNeighbors object
knn = NearestNeighbors(n_neighbors=1)
knn.fit(data.loc[control_indices, ['propensity_score']]) # Fit on control propensity scores

# Find nearest neighbors for treated units
distances, indices = knn.kneighbors(data.loc[treated_indices, ['propensity_score']])

# Match treated units to control units
matched_control_indices = control_indices[indices.flatten()] #indices.flatten() gives the index location of the matching control WITHIN the *control_indices* array.  We need to map this back to the overall dataframe index.

# Create matched dataset
matched_treated = data.loc[treated_indices].copy()
matched_control = data.loc[matched_control_indices].copy()

# Add group identifier
matched_treated['group'] = 'treated'
matched_control['group'] = 'control'

# Combine matched datasets
matched_data = pd.concat([matched_treated, matched_control])

# 3. Outcome Comparison
treated_outcome = matched_data[matched_data['group'] == 'treated']['outcome']
control_outcome = matched_data[matched_data['group'] == 'control']['outcome']

# Calculate Average Treatment Effect on the Treated (ATT)
ATT = treated_outcome.mean() - control_outcome.mean()
print(f"Average Treatment Effect on the Treated (ATT): {ATT}")

# 4. Balance Check (Example: Compare means of 'age' in matched groups)
treated_age_mean = matched_data[matched_data['group'] == 'treated']['age'].mean()
control_age_mean = matched_data[matched_data['group'] == 'control']['age'].mean()
print(f"Mean age in treated group: {treated_age_mean}")
print(f"Mean age in control group: {control_age_mean}")

# Optional: Classification Report of Propensity Score Model Performance
y_pred_test = logistic_model.predict(X_test)
print("\nClassification Report on the Test Data:")
print(classification_report(y_test, y_pred_test))
```

**Explanation:**

1.  **Data Generation:**  Synthetic data is created with covariates (`age`, `education`, `income_prior`), treatment assignment (`treatment`), and an outcome variable (`outcome`).  The treatment assignment is influenced by the covariates, creating confounding.
2.  **Propensity Score Estimation:** A logistic regression model is trained to predict the probability of treatment given the covariates.
3.  **Matching:** Nearest neighbor matching is used to find the closest control unit for each treated unit based on their propensity scores.
4.  **Outcome Comparison:**  The average treatment effect on the treated (ATT) is calculated as the difference in the average outcome between the matched treated and control groups.
5.  **Balance Check:**  The means of the 'age' covariate are compared between the matched groups to assess balance. A significant difference would suggest that balance has not been achieved. A classification report of the propensity score model is included to assess its performance.

**Important Considerations:**

*   This is a simplified example.  Real-world applications require more careful consideration of covariate selection, propensity score model specification, matching algorithm selection, and balance checking.
*   Always check the overlap assumption (common support). You may need to trim or weight observations with extreme propensity scores.
*   Explore more advanced matching algorithms like optimal matching or caliper matching.

## 4) Follow-up question

What are some strategies to address the common support (overlap) assumption in PSM if I observe very few or no control units having propensity scores similar to treated units (or vice versa), especially in specific regions of the covariate space? How can I identify these regions?
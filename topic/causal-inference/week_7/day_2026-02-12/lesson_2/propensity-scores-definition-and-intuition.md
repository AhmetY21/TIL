---
title: "Propensity Scores: Definition and Intuition"
date: "2026-02-12"
week: 7
lesson: 2
slug: "propensity-scores-definition-and-intuition"
---

# Topic: Propensity Scores: Definition and Intuition

## 1) Formal definition (what is it, and how can we use it?)

The propensity score is the conditional probability of assignment to a particular treatment or intervention given a set of observed covariates. In simpler terms, it's the probability that a person receives treatment *given* their characteristics.

Formally, let:

*   T be a binary treatment indicator (1 = treated, 0 = control).
*   X be a set of observed pre-treatment covariates.

Then, the propensity score, denoted as *e(X)*, is defined as:

*   *e(X) = P(T = 1 | X)*

**How can we use it?**

The propensity score allows us to balance the observed covariates between the treatment and control groups, even when there's no randomization.  It addresses confounding by summarizing the influence of multiple covariates into a single scalar value.  This single value can then be used for various causal inference techniques, including:

*   **Propensity Score Matching:**  Match treated individuals with untreated individuals who have similar propensity scores. This creates groups that are more comparable on the observed covariates.
*   **Propensity Score Weighting:**  Weight individuals by the inverse of their propensity score (Inverse Probability of Treatment Weighting or IPTW). This effectively creates a pseudo-population where treatment assignment is independent of the observed covariates. For treated individuals, the weight is 1/e(X). For untreated individuals, the weight is 1/(1-e(X)).
*   **Stratification (or Subclassification):**  Divide the sample into strata based on propensity score values.  Within each stratum, treatment and control groups are expected to be more balanced in terms of observed covariates, allowing for estimation of treatment effects within each stratum.
*   **Covariate Adjustment:** Include the propensity score as a covariate in a regression model. This can help adjust for observed confounding when estimating the treatment effect.

The fundamental idea behind using propensity scores is that if individuals with the same propensity score have the same probability of receiving treatment, then within groups defined by the propensity score, treatment assignment is essentially random (conditional on the observed covariates).  This allows us to estimate causal effects under the assumption of *conditional exchangeability* (or *ignorability*):  the potential outcomes are independent of treatment assignment, conditional on the observed covariates X.

## 2) Application scenario

Imagine a study investigating the effect of a new job training program on future earnings.  Participation in the program is *not* randomized.  Individuals self-select into the program, and those who enroll might differ systematically from those who don't. For example, individuals who are more motivated, highly educated, or have more work experience might be more likely to participate.  These characteristics (motivation, education, experience) are *confounders* because they influence both participation in the training program and future earnings.

Without accounting for these confounders, simply comparing the earnings of those who participated in the program to those who didn't would likely give a biased estimate of the program's effect.  The observed difference in earnings might be due to the training program *or* due to the pre-existing differences in the characteristics of the participants.

Propensity score methods can help to address this confounding.  We can use the observed covariates (motivation, education, experience) to estimate the propensity score for each individual - the probability of participating in the training program *given* those characteristics. We can then use the propensity score to balance the observed covariates between the treatment (program participants) and control (non-participants) groups, allowing for a less biased estimate of the program's impact on earnings.  For example, we might match each program participant with a non-participant who has a similar propensity score, thus creating comparable groups.

## 3) Python method (if possible)
```python
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # For binary treatment

# Sample Data (replace with your actual data)
data = pd.DataFrame({
    'treatment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'age': [30, 25, 35, 40, 28, 32, 45, 27, 38, 31],
    'education': [12, 16, 14, 10, 18, 12, 16, 14, 10, 18],
    'income': [50000, 60000, 70000, 40000, 80000, 55000, 75000, 65000, 45000, 85000]
})

# Define the treatment and covariates
T = data['treatment']
X = data[['age', 'education', 'income']]

# Option 1: Logistic Regression (more common for binary treatment)
# Fit a logistic regression model to estimate propensity scores
model = LogisticRegression(random_state=0)
model.fit(X, T)

# Predict propensity scores
propensity_scores = model.predict_proba(X)[:, 1]  # Probability of treatment = 1

# Add propensity scores to the DataFrame
data['propensity_score'] = propensity_scores

print(data)

# Example usage: Matching (very basic)
# This is a very simplified example and requires more sophisticated matching techniques
from sklearn.neighbors import NearestNeighbors
treated_indices = data[data['treatment'] == 1].index
control_indices = data[data['treatment'] == 0].index

treated_ps = data.loc[treated_indices, 'propensity_score'].values.reshape(-1, 1)
control_ps = data.loc[control_indices, 'propensity_score'].values.reshape(-1, 1)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_ps)
distances, indices = nbrs.kneighbors(treated_ps)

matched_control_indices = control_indices[indices.flatten()]

# Now you have a set of matched indices, which allows you to compare the outcome of interest between these two groups.
print("\nMatched Control Indices:", matched_control_indices)
```

**Explanation:**

1.  **Data Preparation:**  We create a sample DataFrame `data` with a binary treatment variable ('treatment') and covariates ('age', 'education', 'income').  Replace this with your own data.
2.  **Model Fitting:**  We use `LogisticRegression` from `sklearn.linear_model` to estimate the propensity scores. The target variable is the treatment indicator, and the predictors are the observed covariates.
3.  **Propensity Score Prediction:**  `model.predict_proba(X)[:, 1]` predicts the probability of being treated (T=1) for each individual, given their covariates. These probabilities are the propensity scores.
4.  **Integration:** Propensity scores are added to the dataframe.
5.  **Matching example:** A very rudimentary one to one matching is implemented for illustration purposes. This is not considered a complete approach. Use the results to compare the outcome of interest.

**Important Notes:**

*   The choice of model (logistic regression, probit, etc.) for estimating propensity scores depends on the nature of the treatment variable. Logistic regression is commonly used for binary treatments.
*   The inclusion of covariates in the propensity score model is crucial. Include all covariates that are thought to influence both the treatment assignment and the outcome.
*   **Overlapping support (common support) is essential.** Make sure there is overlap in the covariate distributions of the treated and untreated groups. If there is no overlap, propensity score methods may not be appropriate. Common support issues should be addressed before matching using methods such as trimming or re-weighting.
*   Matching requires more sophisticated techniques and libraries, especially for large datasets. Consider libraries like `MatchIt` in R (though its concepts apply generally) or the `CausalML` library in python. The example given is for demonstration only.
*   Evaluate the balance achieved by matching/weighting by examining covariate balance before and after adjusting for propensity scores. Check the Standardized Mean Difference (SMD) for each covariate. A SMD less than 0.1 is generally considered good balance.

## 4) Follow-up question

What are the limitations of propensity score methods, and how do we assess the quality of the estimated propensity scores and the balance achieved using these scores? Consider unobserved confounders in your answer.
---
title: "Cross-Fitting and Orthogonalization"
date: "2026-02-16"
week: 8
lesson: 1
slug: "cross-fitting-and-orthogonalization"
---

# Topic: Cross-Fitting and Orthogonalization

## 1) Formal definition (what is it, and how can we use it?)

Cross-fitting and orthogonalization are techniques used in causal inference to address *confounding* and *model selection bias* when estimating causal parameters, such as the average treatment effect (ATE). They help to obtain *consistent* and *asymptotically normal* estimators, even when nuisance functions (e.g., propensity scores, outcome models) are estimated using flexible machine learning methods.

**Orthogonalization (or Neyman Orthogonality)** aims to construct an *estimating equation* whose solution yields the causal parameter of interest. Crucially, this estimating equation is *orthogonal* to nuisance parameters. This means that small errors in the estimation of the nuisance parameters (e.g., the propensity score) do not affect the asymptotic properties of the estimator for the causal parameter. Orthogonalization achieves this by cleverly manipulating the score function, such that the derivative of the estimating equation with respect to the nuisance parameters is zero (or has expectation zero).

**Cross-fitting** is a technique used in conjunction with orthogonalization to further mitigate bias that arises from overfitting when using flexible machine learning algorithms to estimate the nuisance parameters. In cross-fitting, the data is split into K folds. For each fold, the nuisance parameters are estimated using the data from the *other* K-1 folds.  These estimates are then used to construct orthogonal scores for observations in the held-out fold. This ensures that the nuisance parameter estimates used to evaluate the orthogonal score for a given observation are "out-of-sample" and therefore less subject to overfitting bias.

**How can we use it?**

1. **Identify the causal parameter of interest:** Define the causal effect you want to estimate (e.g., ATE, ATT).
2. **Derive an orthogonal estimating equation:**  This involves some mathematical derivation. The goal is to express the estimation problem in terms of a score function that is orthogonal to nuisance parameters.  The resulting estimating equation will typically involve integrals over the distribution of the data.  Because we only have a sample, we will replace those integrals with sample averages. The nuisance parameters that appear in this equation must be estimated using flexible ML models.
3. **Estimate nuisance parameters using machine learning:** Employ machine learning algorithms to estimate the nuisance parameters (e.g., propensity score, outcome models).
4. **Implement cross-fitting:** Split the data into folds, estimate the nuisance parameters in the out-of-sample fashion, and construct the orthogonal scores.
5. **Solve the estimating equation:** Compute the causal parameter estimate by solving the estimating equation with the cross-fitted, orthogonal scores. This often involves taking the sample mean of the orthogonal score and solving for the causal parameter.
6. **Compute standard errors:**  Because the estimator is asymptotically normal, you can estimate standard errors using the sample variance of the orthogonal score.

## 2) Application scenario

A common application is estimating the Average Treatment Effect (ATE) in observational studies. Let's say we want to estimate the effect of a new drug (treatment) on patient recovery, but we only have observational data where patients themselves chose whether to take the drug.

**Scenario:** We have data on patients, including their characteristics (age, gender, medical history), whether they took the drug (treatment indicator), and their recovery outcome.  Because the drug choice was not randomized, the treated and control groups are likely to differ systematically, leading to confounding.  We can use cross-fitting and orthogonalization to address this.

**Steps:**

1.  **Causal Parameter:**  ATE = E[Y(1) - Y(0)], where Y(1) is the potential outcome under treatment and Y(0) is the potential outcome under control.
2.  **Orthogonal Estimating Equation:** A popular orthogonal score function for the ATE in this case is:

    ```
    score(O; θ) = D * (Y - Q(W, 1)) / e(W) - (1 - D) * (Y - Q(W, 0)) / (1 - e(W)) - (Q(W, 1) - Q(W, 0)) - θ
    ```
    where:
    *   O represents the observed data.
    *   θ is the ATE.
    *   D is the treatment indicator (1 if treated, 0 if control).
    *   Y is the outcome.
    *   Q(W, D) is the expected outcome given covariates W and treatment D (outcome model).
    *   e(W) is the propensity score (probability of treatment given covariates W).

    Solving E[score(O; θ)] = 0 for θ gives us an estimate of the ATE.  Note that the "nuisance parameters" here are Q(W, D) and e(W), and they are estimated from the data.  The specific form of the score function makes it orthogonal; intuitively, errors in estimating the propensity score `e(W)` only affect the variance, but not the *bias*, of the ATE estimate (under some regularity conditions).

3.  **Nuisance Parameter Estimation:**  We estimate `Q(W, D)` and `e(W)` using machine learning models, like random forests or gradient boosting.
4.  **Cross-fitting:** Split the data into, say, 5 folds. For each fold *k*:
    *   Train the machine learning models for `Q(W, D)` and `e(W)` using the data from the other 4 folds.
    *   Use these trained models to predict `Q(W, D)` and `e(W)` for the observations in fold *k*.
    *   Calculate the orthogonal score for each observation in fold *k* using these out-of-sample predictions.
5.  **ATE Estimation:** Average the orthogonal scores across all observations in all folds. The ATE estimate is the value of  θ  that makes this average equal to zero (approximately).
6.  **Standard Error Estimation:** Calculate the sample variance of the orthogonal scores across all observations, and use this to construct a confidence interval for the ATE.

## 3) Python method (if possible)

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

def estimate_ate_cross_fit(treatment, outcome, covariates, n_folds=5):
    """
    Estimates the Average Treatment Effect (ATE) using cross-fitting and
    an orthogonal score.

    Args:
        treatment: A numpy array of treatment indicators (1 for treated, 0 for control).
        outcome: A numpy array of outcomes.
        covariates: A numpy array of covariates.
        n_folds: The number of folds for cross-fitting.

    Returns:
        A tuple containing the ATE estimate and its standard error.
    """

    n_samples = len(treatment)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Ensure reproducibility

    propensity_scores = np.zeros(n_samples)
    outcome_predictions_treated = np.zeros(n_samples)
    outcome_predictions_control = np.zeros(n_samples)
    orthogonal_scores = np.zeros(n_samples)

    for train_index, test_index in kf.split(covariates):
        # Split data into train and test for this fold
        treatment_train, treatment_test = treatment[train_index], treatment[test_index]
        outcome_train, outcome_test = outcome[train_index], outcome[test_index]
        covariates_train, covariates_test = covariates[train_index], covariates[test_index]

        # 1. Estimate propensity score (probability of treatment)
        propensity_model = RandomForestClassifier(random_state=42) #Or another model
        propensity_model.fit(covariates_train, treatment_train)
        propensity_scores[test_index] = propensity_model.predict_proba(covariates_test)[:, 1]  # Probability of treatment

        # 2. Estimate outcome models (E[Y|W, D])
        #   - Outcome model for treated (D=1)
        outcome_model_treated = RandomForestRegressor(random_state=42) #Or another model
        outcome_model_treated.fit(covariates_train[treatment_train == 1], outcome_train[treatment_train == 1])
        outcome_predictions_treated[test_index] = outcome_model_treated.predict(covariates_test)

        #   - Outcome model for control (D=0)
        outcome_model_control = RandomForestRegressor(random_state=42) #Or another model
        outcome_model_control.fit(covariates_train[treatment_train == 0], outcome_train[treatment_train == 0])
        outcome_predictions_control[test_index] = outcome_model_control.predict(covariates_test)

        # 3. Calculate Orthogonal Score
        orthogonal_scores[test_index] = (
            treatment_test * (outcome_test - outcome_predictions_treated[test_index]) / propensity_scores[test_index]
            - (1 - treatment_test) * (outcome_test - outcome_predictions_control[test_index]) / (1 - propensity_scores[test_index])
            - (outcome_predictions_treated[test_index] - outcome_predictions_control[test_index])
        )

    # 4. Estimate ATE
    ate_estimate = np.mean(orthogonal_scores)

    # 5. Estimate Standard Error
    ate_std_error = np.std(orthogonal_scores) / np.sqrt(n_samples)

    return ate_estimate, ate_std_error

# Example usage:
np.random.seed(0) #Ensure reproducibility

n_samples = 200
covariates = np.random.rand(n_samples, 5) #Some simulated covariates
treatment = np.random.randint(0, 2, n_samples) #Simulated treatment
outcome = 2 * treatment + np.sum(covariates, axis=1) + np.random.randn(n_samples) #Simulated outcome

ate, se = estimate_ate_cross_fit(treatment, outcome, covariates)
print(f"ATE Estimate: {ate:.3f}")
print(f"Standard Error: {se:.3f}")
```

**Explanation:**

*   The code implements the steps described in the application scenario.
*   It uses `sklearn` for machine learning models (Random Forest in this example, but other models can be used).
*   `KFold` is used for cross-fitting.
*   The orthogonal score is calculated for each observation within each fold.
*   The ATE is estimated as the mean of the orthogonal scores.
*   The standard error is estimated using the sample variance of the scores.
*   A simple example shows how to use the function.

**Important Considerations:**

*   **Model Choice:**  The choice of machine learning models for the propensity score and outcome models is crucial.  Consider using models that are flexible enough to capture complex relationships but also avoid overfitting.  Regularization is often helpful.
*   **Positivity/Overlap:** Cross-fitting and Orthogonalization will still fail if the positivity/overlap assumption is violated.  The propensity score must be bounded away from 0 and 1 for all individuals in the sample.
*   **Regularity Conditions:**  The theoretical guarantees of consistency and asymptotic normality rely on certain regularity conditions, such as smoothness of the nuisance functions and sufficient sample size.

## 4) Follow-up question

How does cross-fitting with orthogonalization relate to Double Machine Learning (DML) and Targeted Maximum Likelihood Estimation (TMLE)? What are the advantages/disadvantages of each?
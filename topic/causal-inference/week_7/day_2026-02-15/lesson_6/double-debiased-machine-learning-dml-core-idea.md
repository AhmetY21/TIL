---
title: "Double / Debiased Machine Learning (DML) Core Idea"
date: "2026-02-15"
week: 7
lesson: 6
slug: "double-debiased-machine-learning-dml-core-idea"
---

# Topic: Double / Debiased Machine Learning (DML) Core Idea

## 1) Formal definition (what is it, and how can we use it?)

Double/Debiased Machine Learning (DML) is a method for estimating the causal effect of a treatment (or intervention) on an outcome variable, even in the presence of confounding. The "double" part refers to the fact that we use machine learning *twice*: once to predict the outcome variable and once to predict the treatment variable. The "debiased" part refers to the fact that DML provides an estimate of the causal effect that is less sensitive to model misspecification than standard regression approaches.

More formally, DML addresses the problem of estimating the Average Treatment Effect (ATE), often denoted as E[Y(1) - Y(0)], where Y(1) is the potential outcome if everyone received the treatment and Y(0) is the potential outcome if nobody received the treatment.  The key idea is to orthogonalize the treatment variable and the outcome variable with respect to the confounders.  This means we want to remove the influence of the confounders from both the treatment and the outcome before estimating the causal effect.

Here's a breakdown of the DML procedure, assuming we want to estimate the ATE:

1.  **Stage 1:  Nuisance Function Estimation**
    *   Model the outcome variable *Y* as a function of the confounders *X*:  E[Y|X] = m(X). We estimate this using machine learning (e.g., random forest, gradient boosting). Let the prediction be m_hat(X).
    *   Model the treatment variable *T* (binary or continuous) as a function of the confounders *X*: E[T|X] = g(X).  We estimate this using machine learning. Let the prediction be g_hat(X).  If T is binary, this is a propensity score model.

2.  **Stage 2:  Orthogonalization and Causal Effect Estimation**
    *   Calculate the residuals for the outcome variable:  Y_tilde = Y - m_hat(X).
    *   Calculate the residuals for the treatment variable:  T_tilde = T - g_hat(X).
    *   Estimate the causal effect (ATE) by regressing Y_tilde on T_tilde:

    ATE_hat = argmin_beta Σ (Y_tilde - beta * T_tilde)^2.  In other words,  ATE_hat = Cov(Y_tilde, T_tilde) / Var(T_tilde).

DML leverages sample splitting (or cross-fitting) to avoid overfitting and bias. This means we split the data into multiple folds. We use one fold to train the machine learning models (m_hat and g_hat) and then use the *other* folds to compute the residuals (Y_tilde and T_tilde) and estimate the ATE.  This process is repeated for each fold, and the final ATE estimate is the average of the estimates obtained from each fold.

Key benefits of DML:

*   **Robustness to Model Misspecification:**  DML provides consistent estimates of the causal effect even if either the outcome model (m(X)) or the treatment model (g(X)) is misspecified, as long as *one* of them is correctly specified.
*   **Use of Flexible Machine Learning Models:** DML allows for the use of powerful machine learning algorithms to model the complex relationships between confounders, treatment, and outcome.
*   **Asymptotic Normality:** Under certain regularity conditions, the DML estimator is asymptotically normal, allowing for the construction of confidence intervals and hypothesis testing.

## 2) Application scenario

Imagine you are working for a company that offers online advertising. You want to estimate the causal effect of showing users a specific type of advertisement (the treatment, T) on their subsequent purchase behavior (the outcome, Y).  However, users are shown different advertisements based on their demographics, browsing history, and past purchase behavior (the confounders, X).  Simply comparing the purchase rates of users who saw the advertisement versus those who didn't will likely lead to biased results because of these confounding factors.

Using DML, you can:

1.  **Model the relationship between user characteristics (X) and the probability of being shown the advertisement (T) using a machine learning model (g(X)).** This captures the selection bias inherent in how advertisements are targeted.

2.  **Model the relationship between user characteristics (X) and their purchase behavior (Y) using a separate machine learning model (m(X)).** This captures the baseline purchase behavior of users, independent of the advertisement.

3.  **Calculate the residuals Y_tilde and T_tilde.** These residuals represent the variation in purchase behavior and advertisement exposure that is *not* explained by the user characteristics.

4.  **Regress Y_tilde on T_tilde to estimate the causal effect of the advertisement on purchase behavior.** This isolates the causal effect by removing the influence of the confounders.

DML would be beneficial in this scenario because:

*   The relationship between user characteristics and both advertisement exposure and purchase behavior may be complex and non-linear, making it difficult to model using standard linear regression. Machine learning models can capture these complex relationships.
*   It's likely that one of your models is close to correct (either the advertisement targeting model or the purchase prediction model). DML is consistent as long as one is correct.
*   The insights gained can be used to optimize advertisement targeting strategies and improve the return on investment of advertising campaigns.

## 3) Python method (if possible)

While there isn't a single function called "DML" in a common Python library, you can implement DML using existing libraries like `sklearn`, `statsmodels`, and libraries specifically designed for causal inference such as `econml` or `causalml`. Here's an example using `sklearn` and `statsmodels` to illustrate the core idea. For a more robust and efficient implementation, consider using `econml` which provides pre-built DML estimators.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def dml_estimate(Y, T, X, model_Y, model_T, n_folds=2):
    """
    Estimates the Average Treatment Effect (ATE) using Double Machine Learning.

    Args:
        Y: Outcome variable (numpy array or pandas Series).
        T: Treatment variable (numpy array or pandas Series, binary or continuous).
        X: Confounders (numpy array or pandas DataFrame).
        model_Y: Scikit-learn regressor for Y ~ X (e.g., RandomForestRegressor).
        model_T: Scikit-learn classifier/regressor for T ~ X (e.g., RandomForestClassifier if T is binary, RandomForestRegressor if T is continuous).
        n_folds: Number of folds for cross-fitting.

    Returns:
        ate_estimate: Estimated Average Treatment Effect.
    """
    n = len(Y)
    folds = np.array_split(np.random.permutation(n), n_folds)
    ate_estimates = []

    for i in range(n_folds):
        # Split data into training and testing sets for this fold
        train_idx = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        test_idx = folds[i]

        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]

        # Stage 1: Nuisance Function Estimation
        model_Y.fit(X_train, Y_train)
        m_hat = model_Y.predict(X_test)

        model_T.fit(X_train, T_train)
        g_hat = model_T.predict(X_test)

        # Stage 2: Orthogonalization and Causal Effect Estimation
        Y_tilde = Y_test - m_hat
        T_tilde = T_test - g_hat

        # ATE estimation (regression of Y_tilde on T_tilde)
        model = LinearRegression()
        model.fit(T_tilde.reshape(-1, 1), Y_tilde)  # Reshape for single feature
        ate = model.coef_[0]  # Regression coefficient is the ATE
        ate_estimates.append(ate)

    return np.mean(ate_estimates)


if __name__ == '__main__':
    # Generate some synthetic data
    np.random.seed(0)
    n = 1000
    X1 = np.random.normal(size=n)
    X2 = np.random.normal(size=n)
    X = np.column_stack([X1, X2])
    T = 0.5 * X1 + np.random.normal(size=n, scale=0.5)  # Continuous treatment
    Y = 2 + 1.5 * T + 0.3 * X2 + np.random.normal(size=n, scale=0.5)


    # Define machine learning models
    model_Y = RandomForestRegressor(n_estimators=100, random_state=0)
    model_T = LinearRegression() #T is assumed to be continuous


    # Estimate ATE using DML
    ate_estimate = dml_estimate(Y, T, X, model_Y, model_T)
    print(f"Estimated ATE: {ate_estimate}") #Should be close to 1.5
```

**Explanation:**

*   The `dml_estimate` function takes the outcome (Y), treatment (T), confounders (X), and the desired machine learning models as input.
*   It implements cross-fitting by splitting the data into `n_folds` (default is 2).
*   In each fold, it trains the machine learning models to predict Y and T from X.
*   It calculates the residuals Y_tilde and T_tilde.
*   Finally, it estimates the ATE by regressing Y_tilde on T_tilde.  A simple linear regression here to estimate the coefficient.
*   The function returns the average ATE across all folds.
*   The `if __name__ == '__main__'` block demonstrates how to use the function with synthetic data. The `RandomForestRegressor` is used for modeling Y, and Linear Regression for T since it is continuous in the example. You could replace this with any other sklearn Regressor.

**Important Notes:**

*   **Sample Splitting/Cross-fitting:** Sample splitting or cross-fitting is crucial for debiasing the ATE estimate. Without it, the ATE estimate will be biased.
*   **Choice of Models:** The choice of machine learning models (`model_Y` and `model_T`) depends on the specific problem and the nature of the data. It's important to choose models that are appropriate for the type of outcome and treatment variables (e.g., regression for continuous variables, classification for binary variables).  Consider using cross-validation to select the optimal hyperparameters for these models.
*   **Library Recommendation:**  For real-world applications, consider using the `econml` or `causalml` libraries. These libraries provide more sophisticated DML implementations, including:
    *   Pre-built DML estimators with various options for model selection and regularization.
    *   Support for different types of treatments (binary, continuous, multi-valued).
    *   Methods for estimating treatment effect heterogeneity (i.e., how the treatment effect varies across different subgroups of the population).
*   **Continuous Treatment:** In the above code, T is assumed to be continuous. If T is a binary treatment, use `RandomForestClassifier` or similar classifier for `model_T`.

## 4) Follow-up question

How does DML relate to instrumental variables (IV) regression? What are the advantages and disadvantages of DML compared to IV?
---
title: "Meta-Learners: T-Learner, S-Learner, X-Learner"
date: "2026-02-14"
week: 7
lesson: 5
slug: "meta-learners-t-learner-s-learner-x-learner"
---

# Topic: Meta-Learners: T-Learner, S-Learner, X-Learner

## 1) Formal definition (what is it, and how can we use it?)

Meta-learners are a class of algorithms designed to estimate individual treatment effects (ITE) or conditional average treatment effects (CATE) from observational data by leveraging machine learning models. They don't make strong assumptions about the form of the treatment effect, but instead rely on flexible machine learning algorithms to learn the underlying relationships. The core idea is to decompose the problem of estimating treatment effects into multiple prediction problems that can be solved using standard supervised learning techniques. Three common types of meta-learners are:

*   **T-Learner (Two-Learner):**

    *   **Definition:**  The T-Learner trains *two* separate machine learning models. One model predicts the outcome `Y` for the treated group (where treatment `T=1`), and the other model predicts the outcome `Y` for the control group (where treatment `T=0`).  The individual treatment effect (ITE) is then estimated as the difference between the predictions of these two models for the same individual.
    *   **Formal representation:**
        *   Model 1:  `Y_1 = f_1(X)`, where `f_1` is trained on data where `T=1`.
        *   Model 0:  `Y_0 = f_0(X)`, where `f_0` is trained on data where `T=0`.
        *   ITE(X) = `f_1(X) - f_0(X)`
    *   **How to use it:** Train the two models on the respective treatment and control groups.  To predict the ITE for a new observation `X`, feed `X` to both models and subtract the predictions.

*   **S-Learner (Single-Learner):**

    *   **Definition:** The S-Learner trains a *single* machine learning model to predict the outcome `Y` as a function of both the covariates `X` and the treatment `T`.  The ITE is estimated by predicting the outcome under treatment (`T=1`) and under control (`T=0`) for the same individual and taking the difference.
    *   **Formal representation:**
        *   Model: `Y = f(X, T)`
        *   ITE(X) = `f(X, T=1) - f(X, T=0)`
    *   **How to use it:** Train the model on the entire dataset, including both covariates and treatment assignment.  To predict the ITE for a new observation `X`, feed `(X, T=1)` and `(X, T=0)` to the model and subtract the predictions.

*   **X-Learner (Cross-Learner):**

    *   **Definition:** The X-Learner is a more sophisticated approach that attempts to correct for the biases inherent in the T-Learner and S-Learner, particularly when the treatment and control groups have vastly different sizes.  It operates in two stages:
        1.  **Stage 1 (Similar to T-Learner):** Train two models, `f_1(X)` (treated) and `f_0(X)` (control), to predict the outcome `Y` in each group.
        2.  **Stage 2:**
            *   Impute the *individual treatment effects* on both groups:
                *   Treated group:  `D_0i = Y_i - f_0(X_i)` (the difference between the observed outcome and the predicted outcome under control, based on the control model).
                *   Control group: `D_1i = f_1(X_i) - Y_i` (the difference between the predicted outcome under treatment and the observed outcome, based on the treated model).
            *   Train two new models to predict these *imputed treatment effects* (not the outcome directly):
                *   Model for `D_0i`: `g_0(X)` trained on the treated group (`T=1`).
                *   Model for `D_1i`: `g_1(X)` trained on the control group (`T=0`).
            *   Estimate the ITE as a weighted average of the two models trained on imputed treatment effects. The weights are often the propensity score (probability of treatment) and its complement.
    *   **Formal representation:**
        *   Stage 1: Same as T-learner: `Y_1 = f_1(X)` (T=1), `Y_0 = f_0(X)` (T=0)
        *   Stage 2:
            *   `D_0 = Y - f_0(X)` for T=1
            *   `D_1 = f_1(X) - Y` for T=0
            *   `g_0(X)` trained to predict `D_0`
            *   `g_1(X)` trained to predict `D_1`
            *   `e(X) = P(T=1 | X)` (Propensity score)
            *   `ITE(X) = e(X) * g_1(X) + (1 - e(X)) * g_0(X)`
    *   **How to use it:**  Implement the two stages as described above. Requires careful consideration of model choices and propensity score estimation.

## 2) Application scenario

Consider a marketing campaign where customers were randomly assigned to receive a promotional offer (treatment) or not (control). We want to estimate the individual-level impact of the offer on purchase amount.

*   **T-Learner:** Would train one model to predict purchase amount for those who received the offer and another for those who didn't.  The difference between these two predictions for a given customer would be their estimated treatment effect.  Potentially useful if the response to marketing significantly differs between customer segments.
*   **S-Learner:** Would train a single model to predict purchase amount using customer features and whether they received the offer as inputs. This approach assumes that the impact of customer features on purchase behavior is similar regardless of treatment assignment. A simpler model to implement but potentially less accurate than a T-learner if treatment effects strongly vary across customer types.
*   **X-Learner:** Would first train models predicting purchase amount separately for treated and control groups. Then, it would calculate the difference between observed and predicted outcomes for each customer to impute individual treatment effects. Finally, it would train models predicting these imputed effects based on customer features, weighted by the propensity score (e.g., the probability of receiving the offer based on customer features). This would be particularly useful if some customers are much more likely to receive the offer than others (i.e., there is substantial confounding). X-Learner can provide more robust estimates, especially when treatment and control groups are highly imbalanced or when treatment effects are heterogeneous.

Another scenario:

Imagine estimating the effect of a job training program on individuals' income. We have data on participants and non-participants, along with various demographic and economic features.  The goal is to estimate the impact of the training on each person's income. The X-Learner might perform best here, especially if the treatment and control groups have significantly different characteristics and the effect of the training is expected to vary across individuals.

## 3) Python method (if possible)

The `EconML` library in Python provides implementations of these meta-learners.  Here's a basic example demonstrating how to use the `TLearner`, `SLearner`, and `XLearner` for CATE estimation:

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dml import DML
from econml.cate_interpreters import explain_model_with_shap

# Generate some sample data
np.random.seed(0)
n_samples = 1000
X = np.random.rand(n_samples, 5)  # Covariates
T = np.random.randint(0, 2, n_samples)  # Treatment (0 or 1)
Y = 2 * X[:, 0] + 3 * X[:, 1] + 1 * T + np.random.randn(n_samples)  # Outcome

# Split data into train and test sets
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)


# Define base learners
est = RandomForestRegressor(random_state=42, n_estimators=20)
mod = RandomForestRegressor(random_state=42, n_estimators=20)
clf = RandomForestClassifier(random_state=42) # for propensity score in XLearner when treatment is not randomized. In this example, random treatment is provided, no need to supply a classifier. If treatment wasn't randomized, then a classifier should be provided.

# T-Learner
t_learner = TLearner(models=est)
t_learner.fit(Y_train, T_train, X=X_train)
t_te = t_learner.effect(X_test)

# S-Learner
s_learner = SLearner(overall_model=est)
s_learner.fit(Y_train, T_train, X=X_train)
s_te = s_learner.effect(X_test)

# X-Learner
x_learner = XLearner(models=est, propensity_model=clf) # You need a propensity model if the treatment isn't randomized
x_learner.fit(Y_train, T_train, X=X_train)
x_te = x_learner.effect(X_test)

print("T-Learner ITE estimates:", t_te[:5])
print("S-Learner ITE estimates:", s_te[:5])
print("X-Learner ITE estimates:", x_te[:5])

# Example using DML
dml = DML(model_y=est, model_t=est, model_final=est)  #Double Machine Learning
dml.fit(Y_train, T_train, X=X_train)
dml_te = dml.effect(X_test)

print("DML ITE estimates:", dml_te[:5])

# Example with SHAP interpretation for DML:
# This requires that you have shap installed.

try:
    from econml.cate_interpreters import explain_model_with_shap
    # Explain the model
    te_interpreter = explain_model_with_shap(dml)
    te_values = te_interpreter.explain(X_test)

    print("DML SHAP values:", te_values.shape)  # Output: (number of samples, number of features)
except ImportError:
    print("SHAP is not installed.  Install it to use explainer functionality.")
```

**Explanation:**

*   The code generates synthetic data with a known treatment effect.
*   It splits the data into training and testing sets.
*   It initializes base machine learning models (RandomForestRegressor and RandomForestClassifier). The classifiers are used for the propensity score if treatment is not randomised.
*   It creates instances of `TLearner`, `SLearner`, and `XLearner`, specifying the base learners.
*   It fits each meta-learner to the training data.
*   It estimates the individual treatment effects (ITEs) using the fitted models.
*   It prints the first few ITE estimates from each learner.
*   Also adds a basic example using the double machine learning (DML) method.
*   Demonstrates using SHAP to get feature importance.  This only works on models that are compatible with `shap` (e.g., tree-based models and deep neural networks).

**Important notes:**

*   The choice of base learner can significantly impact the performance of meta-learners. Experiment with different algorithms (e.g., linear models, tree-based models, neural networks) and hyperparameter tuning.
*   For X-Learner, accurately estimating the propensity score is crucial. If the treatment assignment is not randomized, you'll need to use a propensity score model to correct for confounding.
*   Model selection and validation are essential to avoid overfitting and ensure reliable ITE estimates.
*   The quality of the data directly impacts the accuracy of the ITE estimates. Address missing data, outliers, and potential biases.

## 4) Follow-up question

How do these meta-learners relate to and differ from other methods for causal inference, such as propensity score matching or instrumental variables? What are the trade-offs involved in choosing a meta-learner approach versus these other methods?
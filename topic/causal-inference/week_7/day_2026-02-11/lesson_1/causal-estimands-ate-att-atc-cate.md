---
title: "Causal Estimands: ATE, ATT, ATC, CATE"
date: "2026-02-11"
week: 7
lesson: 1
slug: "causal-estimands-ate-att-atc-cate"
---

# Topic: Causal Estimands: ATE, ATT, ATC, CATE

## 1) Formal definition (what is it, and how can we use it?)

These estimands are different ways of quantifying the causal effect of a treatment on an outcome. The core idea revolves around comparing potential outcomes – what would have happened if an individual *had* received the treatment versus what would have happened if they *had not* received the treatment. We observe only one of these potential outcomes for each individual in the real world (fundamental problem of causal inference). Estimands are the population-level quantities we aim to estimate from observed data.

*   **ATE (Average Treatment Effect):** The average difference in the outcome if everyone in the population received the treatment versus if no one received the treatment. Mathematically:  `ATE = E[Y(1) - Y(0)]`, where Y(1) is the potential outcome under treatment and Y(0) is the potential outcome under no treatment, and E[] denotes the expected value.  It tells us the overall effect of the treatment on the entire population.

*   **ATT (Average Treatment Effect on the Treated):** The average difference in the outcome for *those who actually received the treatment*, comparing their outcome to what would have happened had they not received the treatment. Mathematically: `ATT = E[Y(1) - Y(0) | T=1]`, where T=1 indicates that the individual received treatment. It's useful when you want to know the effect of the treatment *specifically* on the group that received it.

*   **ATC (Average Treatment Effect on the Control/Untreated):** The average difference in the outcome for *those who did not receive the treatment*, comparing what would have happened had they received the treatment to their actual outcome. Mathematically: `ATC = E[Y(1) - Y(0) | T=0]`, where T=0 indicates that the individual did not receive treatment.  It's useful when you want to know what would have happened if the control group *had* received the treatment.

*   **CATE (Conditional Average Treatment Effect):** The average treatment effect conditional on a specific set of covariates (features). Mathematically: `CATE(x) = E[Y(1) - Y(0) | X=x]`, where X=x represents a specific set of covariate values.  CATE helps identify heterogeneous treatment effects - who benefits the *most* (or the *least*) from the treatment based on their characteristics.  It is not a single value but a function of the covariates.

These estimands are essential for:

*   **Decision-making:**  Informing policies and interventions based on their expected impact.
*   **Understanding causal mechanisms:**  Identifying which subgroups benefit most from a treatment.
*   **Evaluating the effectiveness of interventions:** Determining if a treatment is having the desired effect.

## 2) Application scenario

*   **ATE:** A public health organization wants to evaluate the impact of a new vaccination campaign on the overall rate of infection in a population. The ATE will tell them the average reduction in infection rate across the *entire* population, regardless of whether individuals actually received the vaccine or not.

*   **ATT:** A company implements a new training program for its employees. The ATT will tell them how much the training program improved the performance *specifically* of the employees who participated in the training, compared to what their performance would have been had they not participated.

*   **ATC:**  An online retailer experiments with a new personalized recommendation algorithm.  The ATC tells the retailer what would have happened if the users who *didn't* receive the personalized recommendations *had* received them.  This helps understand the missed opportunity on the control group.

*   **CATE:** A marketing team wants to understand how a new advertising campaign affects different customer segments based on their demographics (age, income, location). The CATE will tell them the treatment effect (increase in sales) for each segment of customers, allowing them to target the campaign more effectively.

## 3) Python method (if possible)

While directly calculating potential outcomes is impossible (due to the fundamental problem of causal inference), various methods in Python libraries help *estimate* these estimands, often relying on assumptions like ignorability (no unobserved confounders). Some examples include propensity score matching/weighting, inverse probability of treatment weighting (IPTW), and causal forests.

Here's an example using the `causalml` library to estimate CATE using a causal forest:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.evaluation import plot_gain

# Generate synthetic data (replace with your actual data)
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5)  # 5 features
T = np.random.randint(0, 2, n_samples)  # Treatment (0 or 1)
Y = 2 * X[:, 0] + T + np.random.randn(n_samples) * 0.5  # Outcome

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['treatment'] = T
df['outcome'] = Y


# Split data into training and testing sets
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    df.drop(columns=['outcome', 'treatment']),
    df['outcome'],
    df['treatment'],
    test_size=0.2,
    random_state=42
)

# Train a UpliftRandomForestClassifier to estimate CATE
uplift_model = UpliftRandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
uplift_model.fit(X_train.values, y_train.values, treat_train.values)

# Predict the CATE on the test set (this gives you the individual treatment effects)
cate_predictions = uplift_model.predict(X_test.values)

# Evaluate the model
plot_gain(y_test, treat_test, cate_predictions)


# Calculate the ATE, ATT, and ATC using the predictions (APPROXIMATIONS!)
treated_indices = treat_test == 1
untreated_indices = treat_test == 0

ate_estimate = np.mean(cate_predictions)
att_estimate = np.mean(cate_predictions[treated_indices])
atc_estimate = np.mean(cate_predictions[untreated_indices])


print(f"Estimated ATE: {ate_estimate}")
print(f"Estimated ATT: {att_estimate}")
print(f"Estimated ATC: {atc_estimate}")
```

**Explanation:**

1.  **Synthetic Data:**  We create synthetic data with features (X), treatment (T), and outcome (Y).  The outcome is designed to have a causal relationship with the treatment and feature 0.
2.  **Data Splitting:**  Split the data into training and test sets to evaluate the model's performance on unseen data.
3.  **`UpliftRandomForestClassifier`:** This is a causal forest implementation from the `causalml` library designed to estimate CATE.
4.  **`fit`:** The model is trained on the training data (features, outcome, and treatment).
5.  **`predict`:** The trained model predicts the CATE for each individual in the test set. This `cate_predictions` array contains an *estimate* of `E[Y(1) - Y(0) | X=x]` for each individual in the test set.
6.  **Evaluation:**  The `plot_gain` function provides a visualization of the model's performance in terms of uplift.

**Important Considerations:**

*   The ATE, ATT, and ATC calculations in the example are *approximations* based on the CATE predictions.  They're not the same as estimating these quantities directly using methods like propensity score weighting.
*   **Assumptions are crucial!** The validity of these estimates relies heavily on the assumptions made by the chosen causal inference method (e.g., ignorability, positivity/overlap).
*   This example uses a causal forest, but other methods (e.g., propensity score matching) can be used depending on the context and assumptions you're willing to make.
*   The `causalml` library has other functionalities to help estimate ATE, ATT and ATC through different methods as well.

## 4) Follow-up question

How do the assumptions required for valid estimation of each of these estimands (ATE, ATT, ATC, and CATE) differ, and how do these differences influence the choice of estimation method in practice?
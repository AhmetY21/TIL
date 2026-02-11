---
title: "Identification vs Estimation vs Inference"
date: "2026-02-11"
week: 7
lesson: 2
slug: "identification-vs-estimation-vs-inference"
---

# Topic: Identification vs Estimation vs Inference

## 1) Formal definition (what is it, and how can we use it?)

In causal inference, identification, estimation, and inference are distinct but related stages in determining a causal effect. They can be thought of as answering the following questions, in order:

*   **Identification:** *Can* we even *theoretically* learn the causal effect from the available data, even with infinite data? This stage focuses on establishing a causal estimand – a mathematical expression representing the causal effect of interest – that can be expressed in terms of the observed data distribution. This often involves making assumptions about the causal structure, such as no unobserved confounders, which are encoded in causal diagrams (DAGs). Without identification, no amount of data or sophisticated estimation techniques can recover the true causal effect. We use causal diagrams and causal calculus (do-calculus) to determine identifiability. A causal effect is *identified* if it can be written as a functional of the observed data distribution.

*   **Estimation:** *How* do we *estimate* the identified causal estimand from a *finite* sample of data? This stage deals with selecting an appropriate statistical estimator and applying it to the data to obtain an estimate of the causal effect. There are many possible estimators, such as regression adjustment, propensity score matching, inverse probability weighting (IPW), and targeted maximum likelihood estimation (TMLE). The choice of estimator depends on factors such as the size and distribution of the data, the complexity of the causal model, and computational constraints.

*   **Inference:** *How confident* are we in our *estimated* causal effect? This stage focuses on quantifying the uncertainty associated with the causal effect estimate. We use statistical techniques, such as calculating standard errors, confidence intervals, and p-values, to assess the precision and reliability of the estimate. Inference helps us determine whether the observed effect is statistically significant and how likely it is to generalize to other populations or settings.  Essentially, we are performing hypothesis testing concerning our estimate.

In summary:

*   **Identification:** Establishes the *theoretical possibility* of learning the causal effect.
*   **Estimation:** Provides a *numerical estimate* of the causal effect from the data.
*   **Inference:** Quantifies the *uncertainty* surrounding the estimated causal effect.

## 2) Application scenario

Imagine we want to understand the causal effect of a new drug (treatment *T*) on patient recovery (*Y*). However, patients who are sicker might be more likely to receive the drug, and their underlying health condition (*X*) also affects their recovery. This is a confounding scenario.

1.  **Identification:** We draw a causal diagram showing that *X* influences both *T* and *Y*. This represents confounding. To identify the causal effect of *T* on *Y*, we might assume that *X* is the only confounder (no unobserved confounding). Then, using do-calculus or other identification techniques, we can express the causal effect as P(Y | do(T)) =  ∑ₓ P(Y | T, X)P(X). This indicates that we can adjust for *X* to remove the confounding bias.

2.  **Estimation:** We have patient data with information on *T*, *Y*, and *X*. We choose an estimator to estimate ∑ₓ P(Y | T, X)P(X). One option is regression adjustment. We build a regression model that predicts *Y* from *T* and *X*. Then, we use this model to predict *Y* for different values of *T* while holding *X* at its observed distribution to implement the formula. We then average the predicted values of Y given each possible intervention on T. Another option is inverse probability of treatment weighting (IPTW) and fit a model P(T|X) and then weight each observation by 1/P(T|X).

3.  **Inference:** After obtaining the estimated causal effect from the chosen estimator (e.g., regression adjustment), we want to understand how reliable it is. We calculate the standard error of the estimate. If the confidence interval is small and does not include zero, we have evidence that the drug has a statistically significant causal effect on recovery.

## 3) Python method (if possible)

The `dowhy` library provides tools for causal inference, enabling identification, estimation, and inference.

```python
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.normal(size=n_samples)
T = np.random.binomial(1, 0.5 + 0.2 * X)  # T is affected by X
Y = 2*T + X + np.random.normal(size=n_samples) # Y is affected by T and X
data = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

# 1. Create a causal model
model = CausalModel(
    data=data,
    treatment='T',
    outcome='Y',
    common_causes=['X'] # X is the observed confounder
)

# 2. Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True) #proceed_when_unidentifiable is useful when demonstrating identification without requiring perfect conditions in this example

print(identified_estimand)

# 3. Estimate the causal effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression", # Backdoor adjustment with linear regression
)

print(estimate)

# 4. Perform inference
refute_results = model.refute_estimate(estimate, method_name="random_common_cause") #Use common refutation methods like adding random common cause

print(refute_results)
```

**Explanation:**

1.  **Causal Model:** We define the causal model, specifying the treatment, outcome, and common causes (confounders).  The DAG structure is implicitly specified using common_causes, instrumental_variables, etc.

2.  **Identification:** The `identify_effect()` function uses the causal graph and do-calculus (or other identification strategies) to determine a suitable causal estimand.  `proceed_when_unidentifiable` forces the library to proceed even if the causal effect can't be perfectly identified. In real applications, it's crucial to ensure identification is correct using background knowledge and causal diagrams.

3.  **Estimation:** The `estimate_effect()` function estimates the causal effect using a specified method (e.g., backdoor adjustment with linear regression). `method_name` specifies the estimator.

4.  **Refutation/Inference:** The `refute_estimate()` function provides sensitivity analysis to address unobserved confounding.

## 4) Follow-up question

How do different causal identification assumptions impact the choice of estimation methods and the validity of causal inference? For example, how would using a different causal structure (e.g., including instrumental variables or mediators) change the estimation strategy?
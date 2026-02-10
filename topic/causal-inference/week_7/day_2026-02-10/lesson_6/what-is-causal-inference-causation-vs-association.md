---
title: "What is Causal Inference? (Causation vs Association)"
date: "2026-02-10"
week: 7
lesson: 6
slug: "what-is-causal-inference-causation-vs-association"
---

# Topic: What is Causal Inference? (Causation vs Association)

## 1) Formal definition (what is it, and how can we use it?)

Causal inference is the process of determining the *causes* of effects. It goes beyond simply observing associations or correlations between variables. While association tells us that two things tend to occur together, causation asserts that one thing *directly influences* another.  We use causal inference to understand how changing one variable will affect another, potentially allowing us to design interventions to achieve desired outcomes.  It's crucial in fields like medicine, economics, and social sciences where we want to understand and influence real-world phenomena.

More formally:

* **Association (Correlation):** Two variables are associated if a change in one is statistically related to a change in the other.  This relationship can be positive, negative, or non-monotonic. It doesn't imply one variable influences the other.  Observed associations could be due to confounding factors (lurking variables), reverse causation, or pure chance.

* **Causation:** A variable (cause) *directly* influences another variable (effect).  A change in the cause *leads to* a change in the effect, and this relationship is not merely coincidental. We are specifically interested in *identifying* and *quantifying* the causal effect of the intervention (the cause) on the outcome (the effect).

The core problem in causal inference is that we can't directly observe the effect of a cause on an individual simultaneously both when the cause is present and when it is absent. This is called the Fundamental Problem of Causal Inference.  Causal inference methods aim to address this problem by employing techniques to estimate causal effects from observational data or through carefully designed experiments.

Key uses of causal inference include:

*   **Predicting the effects of interventions:**  What will happen if we implement a new policy?
*   **Identifying causal risk factors:**  What factors are directly contributing to a disease?
*   **Evaluating the effectiveness of treatments:**  Did the treatment actually cause the improvement?
*   **Designing optimal strategies:** Which intervention strategy will lead to the best outcome?

## 2) Application scenario

Consider a scenario where a company observes a strong positive correlation between ice cream sales and crime rates.  Observational data shows that when ice cream sales increase, crime rates also tend to increase.

**Association (Correlation):** A naive interpretation might suggest that ice cream consumption causes crime, or vice versa (though this is unlikely).

**Causal Inference:**  Causal inference pushes us to look for confounding variables. A more likely explanation is that both ice cream sales and crime rates are influenced by a third variable: **temperature**. Hotter weather leads to increased ice cream sales and also, potentially, increased outdoor activity (including criminal activity). The correlation between ice cream sales and crime rates is *spurious*, not causal.  Intervening to reduce ice cream sales would not likely reduce crime rates. Instead, the causal inference approach would involve identifying interventions that target actual causes of crime (e.g., improving economic conditions, increasing police presence) or recognizing that, in the summer, both will naturally be higher.

## 3) Python method (if possible)

While there isn't a single "causal inference" function in Python that solves all problems, several libraries provide tools and methods for causal inference.  One popular library is `DoWhy`. Here's an example of how `DoWhy` can be used to address a simple causal inference problem using the example above:

```python
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 100

# Temperature (confounder)
temperature = np.random.normal(25, 5, n_samples)

# Ice cream sales (treatment)
ice_cream_sales = 5 + 0.5 * temperature + np.random.normal(0, 2, n_samples)

# Crime rate (outcome)
crime_rate = 10 + 0.8 * temperature + np.random.normal(0, 3, n_samples)

data = pd.DataFrame({'temperature': temperature,
                     'ice_cream_sales': ice_cream_sales,
                     'crime_rate': crime_rate})


# 1. Model the causal graph
model= CausalModel(
        data=data,
        treatment='ice_cream_sales',
        outcome='crime_rate',
        graph="graph[directed=True];temperature->ice_cream_sales;temperature->crime_rate;ice_cream_sales->crime_rate",
        proceed_when_unidentifiable=True)


# 2. Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# 3. Estimate the causal effect
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression")
print(estimate)

# 4. Refute the estimate (sensitivity analysis)
refute_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute_results)
```

Explanation:

1.  **Data Generation:** We create synthetic data that mimics the ice cream sales and crime rate scenario, including the confounding effect of temperature.

2.  **Causal Model:**  We define a `CausalModel` using `DoWhy`.  The key part is the `graph`, which specifies the assumed causal relationships between the variables (temperature influences both ice cream sales and crime rate, and ice cream sales influences crime rate).  This graph represents our prior knowledge or assumptions about the causal structure.

3.  **Identification:**  `model.identify_effect` uses the causal graph to determine a valid causal estimand – a statistical quantity we can estimate from the data to obtain the causal effect.  The `proceed_when_unidentifiable=True` argument forces it to proceed even if the effect cannot be fully identified, enabling exploration.

4.  **Estimation:**  `model.estimate_effect` estimates the causal effect of `ice_cream_sales` on `crime_rate` using the identified estimand and a specified method (here, linear regression with backdoor adjustment to account for the confounder).

5.  **Refutation:** `model.refute_estimate` performs sensitivity analyses to test the robustness of the causal estimate. Here we use "random_common_cause" to try injecting random confounders to see if the estimate is dramatically impacted.

Important: This is a simplified example. Real-world causal inference problems often require more complex modeling, more sophisticated estimation techniques (e.g., matching, instrumental variables), and careful sensitivity analysis.  Also, the causal graph MUST be correct to begin with, or else, the results will be wrong!

## 4) Follow-up question

How do different causal inference methods (e.g., propensity score matching, instrumental variables, regression discontinuity) address confounding variables, and what are the limitations of each method?
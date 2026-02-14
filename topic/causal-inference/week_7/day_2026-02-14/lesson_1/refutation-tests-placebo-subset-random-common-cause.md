---
title: "Refutation Tests (Placebo, Subset, Random Common Cause)"
date: "2026-02-14"
week: 7
lesson: 1
slug: "refutation-tests-placebo-subset-random-common-cause"
---

# Topic: Refutation Tests (Placebo, Subset, Random Common Cause)

## 1) Formal definition (what is it, and how can we use it?)

Refutation tests are methods used in causal inference to assess the robustness of an estimated causal effect. They don't "prove" causality, but they provide evidence that the estimated effect is *likely* causal and not simply due to some other confounding factor or artifact of the data or methodology. The core idea is to create a *counterfactual* scenario where we'd expect the estimated causal effect to disappear if our original estimate is genuinely causal.  If the estimated effect *doesn't* disappear under these refutation scenarios, it raises serious questions about the validity of our initial causal inference.

Here's a breakdown of the specific refutation tests mentioned:

*   **Placebo Treatment Refuter:**  This test substitutes the actual treatment variable with a "placebo" treatment. The placebo treatment should be something that is unrelated to the outcome *except* if there's some other confounder at play. If the original causal effect estimate is robust, we should *not* find a significant effect when using the placebo treatment.  Finding a significant effect with the placebo suggests a hidden confounder or some other source of bias.  We can use this to check if our effect is spurious.

*   **Subset Treatment Refuter:**  This test estimates the causal effect on a *subset* of the data where we *expect* the treatment effect to be smaller or non-existent *based on our understanding of the causal mechanism*. If the original causal effect estimate is truly causal, the effect should be substantially smaller or absent in this subset. The subset needs to be chosen based on some prior knowledge about the causal mechanism.  This can help reveal if the effect is only present for certain populations or situations.

*   **Random Common Cause Refuter:** This test adds a randomly generated variable (a "random common cause") as a potential confounder in our model. The random variable is, by definition, independent of the true causal relationship between treatment and outcome. If the original effect estimate is robust, controlling for this random common cause should *not* substantially change the estimate. A large change suggests that the original estimate was very sensitive to the inclusion of spurious confounders. This allows us to test sensitivity to overfitting.

## 2) Application scenario

Let's consider an example: We're studying the effect of a new job training program (treatment) on income (outcome). We estimate a positive causal effect of the training program on income.

*   **Placebo Treatment Refuter:** We could replace the actual training program with a "placebo program" – say, receiving a brochure about career advice. If the estimated effect of the brochure on income is also significant, it suggests that the original effect we observed might be due to some other factor correlated with *receiving any kind of help*, and not the specific content of the training program itself (e.g., more motivated people seek out both types of help).

*   **Subset Treatment Refuter:**  We might hypothesize that the training program is only effective for people with less than a high-school education. We could estimate the effect of the program *only* on people with a high-school degree or higher. If we still find a substantial positive effect on this subset, it raises doubts about our understanding of the causal mechanism and suggests that the program might benefit a wider range of people or, more worryingly, that the initial estimate was confounded.

*   **Random Common Cause Refuter:** We create a random variable (e.g., drawn from a uniform distribution) and include it as a potential confounder in our causal model. If the estimated effect of the training program on income changes significantly after controlling for this random variable, it suggests that our model is sensitive to including irrelevant variables and might be overfitting, or that our effect is weak and heavily influenced by random fluctuations.

## 3) Python method (if possible)

Here's how you can implement these refutation tests using the `dowhy` library:

```python
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'treatment': np.random.choice([0, 1], size=1000),
    'outcome': np.random.normal(loc=5 + 2*np.random.choice([0, 1], size=1000), scale=1), # Outcome depends on treatment
    'confounder': np.random.normal(loc=0, scale=1, size=1000),
    'subset_variable': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]) #70% 0s, 30% 1s, this is used to create the subset to test the effect on
})

# Create a causal model
model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder']
)

# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate the causal effect
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
print("Original Estimate:", estimate)

# Refutation Tests

# 1. Placebo Treatment Refuter
res_placebo = model.refute_estimate(estimate,
                                     method_name="placebo_treatment_refuter",
                                     placebo_type="permute")  # Or "add_unobserved_common_cause" for a different kind of placebo
print("\nPlacebo Treatment Refuter:", res_placebo)

# 2. Subset Treatment Refuter
subset_condition = "subset_variable==1"  # Adjust to your specific subset
res_subset = model.refute_estimate(estimate,
                                    method_name="subset_refuter",
                                    subset_expression=subset_condition)
print("\nSubset Treatment Refuter:", res_subset)

# 3. Random Common Cause Refuter
res_random = model.refute_estimate(estimate,
                                     method_name="random_common_cause",
                                     num_common_causes=1)
print("\nRandom Common Cause Refuter:", res_random)
```

Key points about the code:

*   We create a `CausalModel` using `dowhy`.
*   We use `identify_effect` and `estimate_effect` to obtain our initial causal estimate.
*   The `refute_estimate` function applies each of the refutation tests.  The `method_name` parameter specifies the type of refutation test to use.
*   For the "placebo_treatment_refuter", the `placebo_type="permute"` option shuffles the treatment values. "add_unobserved_common_cause" adds a new, unrelated confounder.
*   For the "subset_refuter", you specify the `subset_expression` to define the subset of the data to use.
*   For the "random_common_cause",  `num_common_causes` controls the number of random common causes to add.

## 4) Follow-up question

How should I interpret the results of a refutation test that fails (i.e., the refutation test finds a significant effect where it shouldn't or significantly changes the original estimate)? What steps should I take next?
---
title: "Sensitivity Analysis for Unobserved Confounding"
date: "2026-02-13"
week: 7
lesson: 5
slug: "sensitivity-analysis-for-unobserved-confounding"
---

# Topic: Sensitivity Analysis for Unobserved Confounding

## 1) Formal definition (what is it, and how can we use it?)

Sensitivity analysis for unobserved confounding is a technique used in causal inference to assess the robustness of causal effect estimates to the presence of unobserved confounders. In observational studies, we often cannot control for all variables that might influence both the treatment (or exposure) and the outcome, leading to potentially biased causal estimates. Sensitivity analysis aims to quantify how *strong* an unobserved confounder would need to be to overturn or substantially alter the conclusions of our causal analysis. It doesn't eliminate the bias, but it helps us understand how sensitive our results are to violations of the assumption of no unobserved confounding (also known as the ignorability or exchangeability assumption).

Specifically, sensitivity analysis typically involves defining one or more parameters that characterize the relationship between the unobserved confounder, the treatment, and the outcome. These parameters represent the *strength* of confounding. The goal is to then vary these parameters over a plausible range and observe how the estimated causal effect changes. If the estimated effect remains relatively stable (e.g., remains significantly different from zero or doesn't change direction) across a wide range of values for the sensitivity parameters, we can be more confident that our results are robust. Conversely, if a relatively weak unobserved confounder can significantly alter the effect estimate, our conclusions might be fragile.

In essence, we're asking the question: "How much would the unobserved confounder need to be associated with the treatment *and* the outcome to completely explain away the observed effect?" By exploring this question, we gain valuable insight into the credibility of our causal claims.

## 2) Application scenario

Imagine we are studying the effect of a new job training program (treatment) on future earnings (outcome). We observe that individuals who participate in the program earn significantly more than those who don't. However, we are concerned about selection bias: individuals who choose to participate in the program might be more motivated, skilled, or have better social networks than those who don't. We measure and control for several observed confounders like education level, prior work experience, and age.  However, we worry that "motivation" is an unobserved confounder – highly motivated people might be more likely to enroll in the program *and* more likely to have higher future earnings regardless of the program.

In this scenario, we can use sensitivity analysis to assess how strong the unobserved "motivation" variable would need to be, in terms of its relationship with both program participation and future earnings, to explain away the observed positive effect of the training program on earnings. If a plausible level of motivation bias is sufficient to reverse the observed effect, it would strongly suggest that our initial conclusion about the program's effectiveness is unreliable. Conversely, if even a very strong association with motivation cannot eliminate the positive effect, we have more confidence in the program's effectiveness.

Other application scenarios include:

*   Evaluating the effect of a new drug on patient outcomes, accounting for unobserved health behaviors.
*   Assessing the impact of a policy intervention on crime rates, considering unobserved neighborhood characteristics.
*   Determining the influence of a marketing campaign on sales, controlling for unobserved customer preferences.

## 3) Python method (if possible)

While there isn't a single, universally accepted "sensitivity analysis package" in Python that works across all causal inference methods and sensitivity parameterizations, several tools and libraries offer functionalities that can be adapted for sensitivity analysis. Furthermore, many newer packages are being developed to specifically address sensitivity analysis for unobserved confounding. The `statsmodels` package also offers tools that can be used for sensitivity analysis, especially in linear models.

Here's an example using the `EValue` package, which focuses on calculating E-values, a type of sensitivity analysis metric, primarily for relative risks:

```python
import numpy as np
import pandas as pd
from evalue import EValue

# Example data (replace with your actual data)
# In this example, we have a binary treatment, a binary outcome, and no measured covariates,
# to keep the example simple.
# Real-world scenarios would have more data and confounders.
data = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.5, 1000),
    'outcome': np.random.binomial(1, 0.6, 1000) # Slightly positive correlation to treatment
})

# Calculate the observed relative risk (RR)
def calculate_relative_risk(data, treatment_col='treatment', outcome_col='outcome'):
    treated_outcome = data[(data[treatment_col] == 1) & (data[outcome_col] == 1)].shape[0]
    treated_total = data[data[treatment_col] == 1].shape[0]
    untreated_outcome = data[(data[treatment_col] == 0) & (data[outcome_col] == 1)].shape[0]
    untreated_total = data[data[treatment_col] == 0].shape[0]

    rr = (treated_outcome / treated_total) / (untreated_outcome / untreated_total)
    return rr

observed_rr = calculate_relative_risk(data)
print(f"Observed Relative Risk: {observed_rr}")

# Calculate the E-value
evalue_result = EValue.from_relative_risk(observed_rr)
evalue = evalue_result.evalue
print(f"E-value: {evalue}")

# Interpretation: An unmeasured confounder would need to have associations with both the treatment
# and outcome of at least 'evalue' (on the risk ratio scale) to explain away the observed association.
# A larger E-value implies more robust results.

# Calculate the E-value for the lower confidence interval bound (if available).  Not available here without actual confidence intervals calculated initially.
# evalue_lower = EValue.from_relative_risk(confidence_interval_lower_bound) # Placeholder.  You need to calculate CI first.
# print(f"E-value lower: {evalue_lower}")


#Example for Odds Ratio
data['outcome'] = np.random.binomial(1, 0.3, 1000)
def calculate_odds_ratio(data, treatment_col='treatment', outcome_col='outcome'):
    treated_outcome = data[(data[treatment_col] == 1) & (data[outcome_col] == 1)].shape[0]
    treated_nooutcome = data[(data[treatment_col] == 1) & (data[outcome_col] == 0)].shape[0]
    untreated_outcome = data[(data[treatment_col] == 0) & (data[outcome_col] == 1)].shape[0]
    untreated_nooutcome = data[(data[treatment_col] == 0) & (data[outcome_col] == 0)].shape[0]

    oratio = (treated_outcome/treated_nooutcome) / (untreated_outcome/untreated_nooutcome)
    return oratio

observed_oratio = calculate_odds_ratio(data)
print(f"Observed Odds Ratio: {observed_oratio}")
evalue_result_or = EValue.from_odds_ratio(observed_oratio)
evalue_or = evalue_result_or.evalue
print(f"E-value (Odds Ratio): {evalue_or}")



```

**Explanation:**

1.  **Data Generation:** We simulate some example data to have a binary treatment and a binary outcome. **Important**: Replace this with your real data!

2.  **Calculate Observed Relative Risk/Odds Ratio**: We calculate the observed relative risk and odds ratio. This is crucial as the E-value is calculated *based on* the observed effect size.

3.  **`EValue.from_relative_risk()`**: We use the `EValue.from_relative_risk()` or `EValue.from_odds_ratio()` function to calculate the E-value.

4.  **Interpretation**: The E-value represents the minimum strength of association that an unmeasured confounder would need to have with *both* the treatment and the outcome, above and beyond the measured confounders, to explain away the observed association.  Higher E-values suggest more robust evidence. A rough rule of thumb is that E-values above 1.1 are okay, and the greater the E-value, the better.

**Important Considerations:**

*   **Choice of Sensitivity Parameter:** The E-value is just *one* approach to sensitivity analysis. Other approaches involve specifying the correlation or the magnitude of the effect of the unobserved confounder on the treatment and outcome using different metrics.
*   **Plausibility of Parameter Values:** The key to meaningful sensitivity analysis is to choose a plausible range of values for the sensitivity parameters. This often requires domain expertise and careful consideration of the specific context.  The `EValue` package deals with a specific metric (relative risk).
*   **Integration with Causal Inference Methods:** Sensitivity analysis is often performed *after* estimating a causal effect using methods like propensity score matching, inverse probability weighting, or instrumental variables. The sensitivity analysis then assesses the robustness of the estimated effect to unobserved confounding.  It's a supplementary analysis, not a replacement for these methods.

**Other Potential Python tools/approaches:**

*   **`statsmodels`:** You can use `statsmodels` to build a model, calculate coefficients, and then manually implement sensitivity analysis by adding hypothetical confounders and re-estimating the model. This requires a more manual approach but provides flexibility.
*   **DoWhy + Sensitivity Analysis Libraries (in development):** The `DoWhy` library is a popular tool for causal inference.  It is currently developing or integrating with libraries focusing on sensitivity analysis to provide more integrated solutions. Watch for updates in the DoWhy ecosystem.
*   **R's `sensitivitymv` (use `rpy2`):** The `sensitivitymv` package in R provides a general framework for sensitivity analysis, including the ability to plot sensitivity contours.  You can use `rpy2` in Python to call R functions and access this package.
*   **Roll your own**: Sensitivity analyses can be conducted even without advanced tools, e.g., you could try to establish a range of plausible confounder magnitudes using domain knowledge and calculate the resulting bias.

## 4) Follow-up question

How do you decide on a "plausible range" for the sensitivity parameters in a real-world application? What factors should you consider? Are there any guidelines or best practices for selecting this range? This is important because if the plausible range is too narrow, the sensitivity analysis is meaningless, and if it is too wide, it becomes uninformative.
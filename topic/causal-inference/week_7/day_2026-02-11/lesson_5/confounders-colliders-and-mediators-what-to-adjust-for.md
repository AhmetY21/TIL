---
title: "Confounders, Colliders, and Mediators (What to Adjust For?)"
date: "2026-02-11"
week: 7
lesson: 5
slug: "confounders-colliders-and-mediators-what-to-adjust-for"
---

# Topic: Confounders, Colliders, and Mediators (What to Adjust For?)

## 1) Formal definition (what is it, and how can we use it?)

In causal inference, understanding the roles of confounders, colliders, and mediators is crucial for accurate causal effect estimation. Identifying and appropriately handling these variables can significantly affect the validity of your conclusions.

*   **Confounder:** A confounder is a variable that is associated with *both* the treatment (or exposure) and the outcome. It distorts the apparent relationship between the treatment and the outcome by creating a spurious association.
    *   Formally, `Z` is a confounder if:
        *   `Z -> X` (Z causes X, where X is the treatment)
        *   `Z -> Y` (Z causes Y, where Y is the outcome)
    *   *How to use it:* To obtain an unbiased estimate of the causal effect of X on Y, you need to *adjust for* or *control for* confounders. This can be done using techniques like regression, matching, stratification, or inverse probability weighting.  Adjusting for a confounder breaks the backdoor path, allowing you to isolate the causal effect of X on Y. Failing to adjust for confounders leads to *confounding bias*.

*   **Collider:** A collider is a variable that is caused by *both* the treatment and the outcome. Conditioning on (i.e., adjusting for) a collider can create a spurious association between the treatment and the outcome, even if no real association exists. This is known as *collider bias* or *Berkson's paradox*.
    *   Formally, `Z` is a collider if:
        *   `X -> Z` (X causes Z)
        *   `Y -> Z` (Y causes Y)
    *   *How to use it:*  You should *avoid* adjusting for colliders. Conditioning on a collider opens a non-causal path between the treatment and the outcome, distorting the true causal effect.

*   **Mediator:** A mediator is a variable that lies on the causal pathway between the treatment and the outcome. The treatment affects the mediator, which in turn affects the outcome.
    *   Formally, `Z` is a mediator if:
        *   `X -> Z` (X causes Z)
        *   `Z -> Y` (Z causes Y)
    *   *How to use it:* Whether to adjust for a mediator depends on the specific causal question you are trying to answer.
        *   If you want to estimate the *total effect* of the treatment on the outcome, you should *not* adjust for the mediator.  The total effect includes the effect mediated through Z.
        *   If you want to estimate the *direct effect* of the treatment on the outcome (i.e., the effect of the treatment on the outcome *not* through the mediator), you should adjust for the mediator. This isolates the effect of X on Y that is independent of Z. Mediation analysis can be used to quantify both the direct and indirect effects.

In summary:

*   Adjust for **confounders**.
*   Don't adjust for **colliders**.
*   Adjust for **mediators** only if you are interested in the direct effect, not the total effect.

## 2) Application scenario

Imagine studying the effect of a new drug (X) on recovery time (Y) from an illness.

*   **Confounder:**  A patient's overall health (Z) before taking the drug might be a confounder.  Healthier patients are more likely to be prescribed the drug *and* more likely to recover faster regardless of the drug.  Failing to account for pre-existing health would lead to a biased estimate of the drug's effectiveness.  We would need to adjust for pre-existing health.

*   **Collider:** Suppose both the drug (X) and the severity of the illness (Y, measured inversely as recovery time) independently influence whether a patient participates in a follow-up study (Z). Participating in the study is then a collider. Adjusting for study participation would create a spurious association between the drug and recovery time.  For example, it might appear that the drug is more effective than it actually is because patients who both took the drug *and* recovered quickly are more likely to participate. Conversely, patients who did not take the drug *and* had a slow recovery might also be selected for the follow up study due to specific case investigations.

*   **Mediator:**  The drug (X) might lower inflammation (Z), which then leads to faster recovery (Y).  Inflammation is a mediator. If we want to know the total effect of the drug on recovery, we shouldn't adjust for inflammation. If we want to know the direct effect of the drug on recovery (i.e., the effect *not* mediated through reduced inflammation), we *should* adjust for inflammation.

## 3) Python method (if possible)

The `causalml` library can be used for mediation analysis and causal effect estimation.  This example focuses on simulating data and using `causalml` to analyze the effects, but emphasizes the roles of confounders, colliders and mediators.

```python
import numpy as np
import pandas as pd
from causalml.mediation import MediationAnalysis
from sklearn.linear_model import LinearRegression

# Simulate data
np.random.seed(42)
n = 1000

# Confounder
confounder = np.random.normal(0, 1, n)

# Treatment
treatment = 0.5 * confounder + np.random.normal(0, 1, n)

# Mediator
mediator = 0.7 * treatment + 0.3 * confounder + np.random.normal(0, 1, n)

# Outcome
outcome = 0.4 * treatment + 0.6 * mediator + 0.2 * confounder + np.random.normal(0, 1, n)

# Collider
collider = 0.6 * treatment + 0.4 * outcome + np.random.normal(0, 1, n)

df = pd.DataFrame({'treatment': treatment,
                   'outcome': outcome,
                   'confounder': confounder,
                   'mediator': mediator,
                   'collider': collider})

# Perform mediation analysis using causalml

# Instantiate mediation analysis object
med = MediationAnalysis(
    outcome_model=LinearRegression(),
    treatment_model=LinearRegression(),
    mediator_model=LinearRegression()
)

# Fit the model
med.fit(
    X=df[['treatment']],  # treatment
    M=df[['mediator']],  # mediator
    Y=df['outcome'],      # outcome
    C=df[['confounder']] # confounder
)


# Get results
summary = med.summary()
print(summary)


# Example of using regression to adjust for a confounder
from statsmodels.formula.api import ols
# Adjusting for the confounder
model_adjusted = ols("outcome ~ treatment + confounder", data=df).fit()
print(model_adjusted.summary())

#  Regression without adjusting for confounder (biased result)
model_unadjusted = ols("outcome ~ treatment", data=df).fit()
print(model_unadjusted.summary())

```

This code simulates a scenario with a treatment, outcome, confounder, mediator, and collider. It then performs mediation analysis using `causalml` to estimate direct and indirect effects. Additionally, it shows how adjusting for the confounder in a regression model yields different results than without adjustment. The results highlight the importance of correctly identifying and addressing these variables.  The direct effect will differ depending on whether you include the mediator and confounder in your model.

**Explanation:**

1.  **Data Simulation:**  We create synthetic data representing the relationships between the variables as described in the application scenario.  Note the relationships specified in the variable definitions.
2.  **Mediation Analysis with `causalml`:** `causalml` estimates the Average Causal Mediation Effect (ACME), Average Direct Effect (ADE), and Total Effect. The code specifies the models for each part of the causal chain (treatment, mediator, outcome) and includes the confounder as a control variable (`C`).  The `summary()` method provides insights into these effects.
3.  **Regression for Confounder Adjustment:** The `statsmodels` library is used to demonstrate how adjusting for a confounder can change the estimated effect of the treatment. The `ols` function fits an ordinary least squares regression model. Comparing the coefficients for "treatment" in the adjusted and unadjusted models demonstrates the impact of confounding.  The model without adjustment will be biased.
4.  **Collider demonstration.** If you include the collider in your model, the impact of the treatment on the outcome may diminish because some of the apparent correlation between treatment and outcome may be due to the collider.

## 4) Follow-up question

How can I determine the correct causal relationships between variables in a real-world dataset to accurately identify confounders, colliders, and mediators, especially when I lack extensive domain knowledge?  What tools or methods exist to help visualize and infer these causal structures?
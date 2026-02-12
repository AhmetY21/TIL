---
title: "Standardization and G-Formula (Outcome Modeling)"
date: "2026-02-12"
week: 7
lesson: 1
slug: "standardization-and-g-formula-outcome-modeling"
---

# Topic: Standardization and G-Formula (Outcome Modeling)

## 1) Formal definition (what is it, and how can we use it?)

**Standardization (also known as the G-Formula or Outcome Modeling)** is a method in causal inference used to estimate the average treatment effect (ATE) or other causal effects by simulating what would happen to the population's outcome under different intervention scenarios.  It relies on building a predictive model for the outcome variable (Y) based on observed covariates (X) and the treatment variable (A). This model is then used to predict the outcome under different fixed values or distributions of the treatment variable, averaging over the observed distribution of the covariates.

**How it works:**

1. **Model Building:** First, a regression model is built to predict the outcome (Y) based on the treatment (A) and relevant covariates (X).  This model is typically written as E[Y|A,X] = f(A, X), where f is some function learned from the data.  This model *must* control for confounding; that is, the covariates X must be sufficient to block all backdoor paths from A to Y.  It is also crucial that this model is correctly specified.

2. **Prediction Under Intervention:** After the model is built, we want to simulate a hypothetical intervention, such as "treat everyone."  This involves setting the treatment variable (A) to a fixed value (e.g., A=1 for "treat everyone") for *every individual* in the dataset.

3. **Averaging:** Using the model from step 1 and the hypothetical intervention from step 2, we predict the outcome for each individual in the dataset, but now *as if* they had all received the treatment specified in the intervention. We then average these predicted outcomes to obtain an estimate of the mean outcome under the intervention.

4. **Comparison (for ATE):** To estimate the ATE, we repeat steps 2 and 3 for different treatment values (e.g., A=0 for "treat no one") and then calculate the difference between the average predicted outcomes under each scenario.

**Use cases:**

*   Estimating the ATE of a treatment.
*   Predicting the impact of hypothetical interventions.
*   Estimating direct and indirect effects in mediation analysis (with appropriate adaptations).
*   Dealing with time-varying treatments and confounders (Marginal Structural Models).

**Key Assumption:** Conditional exchangeability (no unmeasured confounding). The model E[Y|A,X] must adequately adjust for all confounding between treatment and outcome.

## 2) Application scenario

**Scenario:**  Suppose we want to estimate the causal effect of a new drug (A=1 if treated, A=0 if not) on blood pressure (Y). We also have data on age (X1), weight (X2), and smoking status (X3). We suspect that age, weight, and smoking are all confounders, meaning they are related to both the decision to take the drug and the blood pressure outcome.

**Without standardization/G-Formula,** simply comparing the average blood pressure between the treated and untreated groups would likely be biased due to these confounders.

**With standardization/G-Formula:**

1.  We would build a regression model predicting blood pressure (Y) based on treatment (A), age (X1), weight (X2), and smoking status (X3):  `E[Y|A, X1, X2, X3] = f(A, X1, X2, X3)`.  This might be a linear regression, a more flexible model like a generalized additive model, or a machine learning model, depending on the complexity of the relationship.

2.  To estimate the effect of treating everyone (A=1), we would then set A=1 for *every individual* in our dataset but keep their observed age, weight, and smoking status. We would then use the model from step 1 to predict each individual's blood pressure *as if* they were all treated.

3.  We would then average these predicted blood pressure values across all individuals.  This would give us an estimate of the average blood pressure in the population *if everyone were treated*.

4.  We would repeat steps 2 and 3, but setting A=0 for everyone.  This gives us an estimate of the average blood pressure in the population *if no one were treated*.

5.  The difference between the average predicted blood pressure when A=1 and the average predicted blood pressure when A=0 gives us an estimate of the ATE of the drug on blood pressure.

## 3) Python method (if possible)

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'A': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Treatment (1=treated, 0=untreated)
    'X1': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],  # Age
    'X2': [150, 160, 170, 180, 190, 200, 210, 220, 230, 240],  # Weight
    'Y': [120, 130, 125, 140, 135, 150, 145, 160, 155, 170]  # Blood Pressure
})

# 1. Build a regression model
model = smf.ols('Y ~ A + X1 + X2', data=data).fit()  # OLS Regression
# Alternatively, use a GLM for different outcome types:
# model = smf.glm('Y ~ A + X1 + X2', data=data, family=sm.families.Gaussian()).fit() # Gaussian for continuous
# model = smf.glm('Y ~ A + X1 + X2', data=data, family=sm.families.Binomial()).fit() # Binomial for binary outcome


# 2. Create hypothetical scenarios (treat everyone and treat no one)
data_treated = data.copy()
data_treated['A'] = 1  # Set everyone to treated

data_untreated = data.copy()
data_untreated['A'] = 0  # Set everyone to untreated

# 3. Predict outcomes under each scenario
predictions_treated = model.predict(data_treated)
predictions_untreated = model.predict(data_untreated)


# 4. Calculate average predicted outcomes
avg_treated = predictions_treated.mean()
avg_untreated = predictions_untreated.mean()

# 5. Calculate the Average Treatment Effect (ATE)
ate = avg_treated - avg_untreated

print(f"Average Predicted Outcome if Everyone is Treated: {avg_treated}")
print(f"Average Predicted Outcome if No One is Treated: {avg_untreated}")
print(f"Estimated Average Treatment Effect (ATE): {ate}")


# More Robust Approach: Bootstrap Confidence Intervals
import numpy as np

def estimate_ate(data, model_formula, n_bootstraps=1000):
    ate_estimates = []
    for _ in range(n_bootstraps):
        # Bootstrap sample with replacement
        bootstrap_sample = data.sample(frac=1, replace=True)

        # Fit model on bootstrap sample
        bootstrap_model = smf.ols(model_formula, data=bootstrap_sample).fit()

        # Create hypothetical datasets
        data_treated = bootstrap_sample.copy()
        data_treated['A'] = 1
        data_untreated = bootstrap_sample.copy()
        data_untreated['A'] = 0

        # Predict and calculate ATE
        predictions_treated = bootstrap_model.predict(data_treated)
        predictions_untreated = bootstrap_model.predict(data_untreated)
        ate_estimates.append(predictions_treated.mean() - predictions_untreated.mean())

    return ate_estimates


# Use the bootstrap function
ate_estimates = estimate_ate(data, 'Y ~ A + X1 + X2')

# Calculate the confidence interval (e.g., 95% CI)
ci_lower = np.percentile(ate_estimates, 2.5)
ci_upper = np.percentile(ate_estimates, 97.5)

print(f"ATE Bootstrap Confidence Interval: ({ci_lower}, {ci_upper})")
```

**Explanation:**

*   The code first creates a sample dataset. Replace this with your actual data.
*   It then builds a linear regression model using `statsmodels`.  **Crucially, the model must include all relevant confounders** (X1 and X2 in this case). This example uses ordinary least squares (OLS), which is appropriate for continuous outcomes.  For binary outcomes, you would use a logistic regression (Generalized Linear Model with a binomial family).
*   It creates two new dataframes, `data_treated` and `data_untreated`, where the treatment variable (A) is set to 1 and 0, respectively, for all individuals.
*   It uses the fitted model to predict the outcome (Y) for each individual under each scenario.
*   It calculates the average predicted outcome for each scenario.
*   Finally, it calculates the ATE by subtracting the average predicted outcome when A=0 from the average predicted outcome when A=1.
* The code also provides a bootstrap approach to calculate confidence intervals, which is generally recommended for more robust inference.

**Important Considerations:**

*   **Model Specification:** The choice of the model (linear, logistic, non-parametric, etc.) is critical.  The model *must* adequately capture the relationship between the treatment, covariates, and outcome.  Model misspecification can lead to biased estimates. Consider using techniques like cross-validation or sensitivity analysis to assess model robustness.
*   **Functional Form:** The functional form of the model (e.g., linear vs. non-linear relationship between age and blood pressure) needs to be appropriately specified. Consider adding interaction terms or using splines to capture non-linearities.
*   **Confounding:** The model *must* control for all relevant confounders. If there is unmeasured confounding, the estimated ATE will be biased.
*   **Positivity (Overlap):** The positivity assumption requires that for every combination of covariates X, there is a non-zero probability of receiving each treatment value. Violation of positivity can lead to unstable or unreliable estimates. Check for regions of covariate space where treatment assignment is highly predictable (e.g., almost everyone with X=x gets treatment A=1).

## 4) Follow-up question

How does standardization/G-formula compare to Inverse Probability of Treatment Weighting (IPTW) in terms of assumptions and robustness?  What are some practical considerations for choosing between the two methods?
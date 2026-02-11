---
title: "Estimating Effects by Covariate Adjustment (Regression as Adjustment)"
date: "2026-02-11"
week: 7
lesson: 6
slug: "estimating-effects-by-covariate-adjustment-regression-as-adjustment"
---

# Topic: Estimating Effects by Covariate Adjustment (Regression as Adjustment)

## 1) Formal definition (what is it, and how can we use it?)

Covariate adjustment, also known as regression as adjustment, is a method used in causal inference to estimate the causal effect of a treatment (or exposure) on an outcome while controlling for confounding variables (also called covariates or confounders).  The fundamental idea is to use regression analysis to statistically remove the association between the confounders and both the treatment and the outcome.  By adjusting for these confounders, we aim to isolate the true causal effect of the treatment.

More formally, suppose we have a treatment variable *T*, an outcome variable *Y*, and a set of covariates *C*. The goal is to estimate the average treatment effect (ATE):

*ATE = E[Y(T=1) - Y(T=0)]*

where Y(T=1) and Y(T=0) represent the potential outcomes if everyone in the population received treatment (T=1) and if no one received treatment (T=0), respectively. We use potential outcomes to clearly define the average treatment effect and what we are aiming to estimate.

Regression as adjustment proceeds by fitting a regression model of the form:

*E[Y | T, C] =  β<sub>0</sub> + β<sub>1</sub>T + β<sup>T</sup>C*

where:

*   *E[Y | T, C]* is the expected value of the outcome *Y* given the treatment *T* and covariates *C*.
*   *β<sub>0</sub>* is the intercept.
*   *β<sub>1</sub>* is the coefficient for the treatment *T*, representing the estimated average treatment effect *after* adjusting for the covariates *C*. This is the key quantity of interest.
*   *β* is a vector of coefficients for the covariates *C*.
*   *C* is a vector of observed confounders.

The core assumption is that *C* blocks all backdoor paths between *T* and *Y* (i.e., satisfies the backdoor criterion).  This means that, conditional on *C*, the treatment *T* is independent of the potential outcomes *Y(T=1)* and *Y(T=0)*.  If this assumption holds, *β<sub>1</sub>* provides an unbiased estimate of the ATE.

We can use this method to estimate the causal effect of a variety of treatments on outcomes.  For instance, the effect of a new drug on patient recovery, the effect of a job training program on future income, or the effect of a new policy on air quality.

## 2) Application scenario

Imagine we want to study the effect of a new online tutoring program (T) on student exam scores (Y). We observe that students who choose to enroll in the tutoring program tend to be higher-achieving students to begin with.  Therefore, simply comparing the exam scores of students who enrolled in the program versus those who didn't would be misleading due to *selection bias*.  Prior academic performance (C), measured by their GPA before enrolling in the program, is a confounding variable. Students with higher GPAs are more likely to enroll in the program and tend to score higher on exams regardless of the program's effectiveness.

To address this confounding, we can use regression as adjustment.  We would fit a regression model with exam score (Y) as the dependent variable and the tutoring program (T) and GPA (C) as independent variables:

*E[Y | T, C] =  β<sub>0</sub> + β<sub>1</sub>T + β<sub>2</sub>C*

The coefficient *β<sub>1</sub>* would then estimate the effect of the tutoring program on exam scores *after* adjusting for the student's prior GPA. If β<sub>1</sub> is positive and statistically significant, we can conclude that the tutoring program has a positive effect on exam scores, even after accounting for differences in pre-existing academic ability. This gives us a more accurate estimate of the program's causal impact.  We are essentially comparing students with similar GPAs, where one student is in the tutoring program and one is not.

## 3) Python method (if possible)

```python
import statsmodels.api as sm
import pandas as pd

# Simulate some data
import numpy as np
np.random.seed(42)  # for reproducibility

n = 1000  # Number of observations
gpa = np.random.normal(3.0, 0.5, n)  # GPA (confounder)
treatment = np.random.binomial(1, (gpa - 2.0) / 3.0, n)  # Treatment assignment (influenced by GPA)
outcome = 50 + 10 * treatment + 15 * gpa + np.random.normal(0, 10, n)  # Outcome (influenced by treatment and GPA)

data = pd.DataFrame({'outcome': outcome, 'treatment': treatment, 'gpa': gpa})

# Fit the regression model
X = data[['treatment', 'gpa']]  # Independent variables (treatment and confounder)
X = sm.add_constant(X)  # Add a constant (intercept)
y = data['outcome']  # Dependent variable (exam score)

model = sm.OLS(y, X)
results = model.fit()

# Print the results
print(results.summary())

# Extract the estimated treatment effect (coefficient of 'treatment')
treatment_effect = results.params['treatment']
print(f"\nEstimated Treatment Effect: {treatment_effect:.2f}")
```

This Python code uses the `statsmodels` library to fit an ordinary least squares (OLS) regression model.  It simulates data reflecting the described scenario, creates a DataFrame, and then fits the model. The `results.summary()` output provides a detailed overview of the regression results, including the estimated coefficients, standard errors, p-values, and R-squared. The `treatment_effect` variable then extracts the crucial coefficient for the 'treatment' variable, representing the estimated causal effect of the online tutoring program on exam scores after adjusting for GPA.

## 4) Follow-up question

What are some limitations of using regression as adjustment for causal inference, and what alternative methods might be more appropriate when those limitations are present?
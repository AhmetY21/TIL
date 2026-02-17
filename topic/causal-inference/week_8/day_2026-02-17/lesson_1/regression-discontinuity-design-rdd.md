---
title: "Regression Discontinuity Design (RDD)"
date: "2026-02-17"
week: 8
lesson: 1
slug: "regression-discontinuity-design-rdd"
---

# Topic: Regression Discontinuity Design (RDD)

## 1) Formal definition (what is it, and how can we use it?)

Regression Discontinuity Design (RDD) is a quasi-experimental method used to estimate the causal effect of an intervention (treatment) when assignment to the treatment is determined by whether an observed variable (the running variable or assignment variable) exceeds a specific threshold (the cutoff).

In essence, RDD leverages the discontinuity in treatment assignment at the cutoff point.  Individuals just below the cutoff receive no treatment (or a different treatment), while individuals just above the cutoff receive the treatment.  The key assumption is that, absent the treatment, individuals on either side of the cutoff would have similar outcomes.  Therefore, any jump in the outcome variable at the cutoff can be attributed to the treatment effect.

Formally, let:

*   `Y_i` be the outcome of interest for individual `i`.
*   `X_i` be the running variable (or assignment variable) for individual `i`.
*   `c` be the cutoff value.
*   `T_i` be the treatment indicator, such that `T_i = 1` if `X_i >= c` and `T_i = 0` if `X_i < c`.

The causal effect is estimated by comparing the predicted outcome just above the cutoff with the predicted outcome just below the cutoff:

`Causal Effect = lim_{x → c+} E[Y_i | X_i = x] - lim_{x → c-} E[Y_i | X_i = x]`

There are two main types of RDD:

*   **Sharp RDD:** Treatment assignment is a deterministic function of the running variable exceeding the cutoff.  Everyone above the cutoff gets treated, and everyone below does not.  This is the most common type.
*   **Fuzzy RDD:** Treatment assignment is probabilistic, rather than deterministic.  Exceeding the cutoff increases the *probability* of receiving treatment, but some individuals above the cutoff may not be treated, and some below the cutoff might be. In this case, the analysis typically involves using the running variable exceeding the cutoff as an instrument for treatment.

We use RDD to estimate the local average treatment effect (LATE) at the cutoff point. The results are generally interpreted as only being applicable for individuals very close to the cutoff.

## 2) Application scenario

A classic example is the effect of receiving a scholarship on academic performance. Suppose a scholarship is awarded to students who score above a certain threshold (the cutoff) on a standardized test (the running variable).

*   **Outcome Variable (Y):** GPA or other measure of academic success.
*   **Running Variable (X):** Score on the standardized test.
*   **Cutoff (c):** The test score required to receive the scholarship.
*   **Treatment (T):** Receiving the scholarship (T = 1) or not (T = 0).

We can compare the GPAs of students who scored just above the cutoff to those who scored just below the cutoff. If there's a significant jump in GPA at the cutoff, it suggests that receiving the scholarship had a positive effect on academic performance.

Other examples include:

*   The effect of incumbency on election outcomes (the cutoff could be 50% of the vote in the previous election).
*   The effect of class size on student achievement (the cutoff could be a maximum class size; if enrollment exceeds it, the class is split).
*   The effect of a minimum drinking age law on traffic fatalities (the cutoff is the legal drinking age).

## 3) Python method (if possible)

While a single, readily available RDD function isn't included in common Python statistical packages like `statsmodels`, we can implement RDD using regression with carefully chosen polynomial terms and interaction terms, as well as the `rdd` package available on PyPI. Here's a demonstration using the `rdd` package.

```python
import numpy as np
import pandas as pd
import rdd

# Generate some synthetic data
np.random.seed(42)
n = 1000
cutoff = 0
X = np.random.normal(0, 1, n)
T = (X >= cutoff).astype(int)  # Treatment assignment
Y_0 = 2 + 0.5 * X + np.random.normal(0, 0.5, n)  # Potential outcome without treatment
tau = 1  # Treatment effect
Y_1 = Y_0 + tau #Potential outcome with treatment
Y = Y_0 + T * tau  # Observed outcome

data = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

# RDD estimation using the rdd package
rdd_object = rdd.rdd(data['Y'], data['X'], cutoff) # Create the RDD object
results = rdd_object.fit() # Perform the regression
print(results) # Print the results
```

**Explanation:**

1.  **Data Generation:**  Simulates data where the outcome `Y` is a function of the running variable `X`, a treatment effect `tau`, and some noise. `T` indicates whether a unit is treated based on if `X` exceeds the cutoff.
2.  **RDD Object:**  Creates an `rdd` object using the outcome, running variable, and cutoff.
3.  **Estimation:**  The `fit()` method performs a local linear regression around the cutoff to estimate the treatment effect. By default the `rdd` package uses local linear regression with triangular weights.
4. **Output:** The results will include estimates of the treatment effect and standard errors. The `fit()` method also returns information such as the bandwidth used for the local regression.
Note that more sophisticated approaches might involve choosing an appropriate bandwidth, using higher-order polynomials, and checking for sensitivity to different bandwidth choices. It is also important to check for manipulation of the running variable around the cutoff to ensure the validity of the analysis.

## 4) Follow-up question

How do you determine the appropriate bandwidth to use in a Regression Discontinuity Design, and how sensitive are the results to different bandwidth choices? What are the potential implications of using a bandwidth that is too wide or too narrow?
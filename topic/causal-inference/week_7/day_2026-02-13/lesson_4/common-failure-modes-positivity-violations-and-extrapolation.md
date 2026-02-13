---
title: "Common Failure Modes: Positivity Violations and Extrapolation"
date: "2026-02-13"
week: 7
lesson: 4
slug: "common-failure-modes-positivity-violations-and-extrapolation"
---

# Topic: Common Failure Modes: Positivity Violations and Extrapolation

## 1) Formal definition (what is it, and how can we use it?)

**Positivity Violation (or Lack of Overlap):** Positivity, also called overlap or common support, refers to the condition that for every value of the confounders, there is a non-zero probability of receiving each treatment level. Formally, for a treatment *A* and covariates *X*, positivity requires that 0 < P(A = a | X = x) < 1 for all values *a* that *A* can take and for all values *x* in the support of *X*.  A positivity violation occurs when this condition is not met. This often manifests as regions of the covariate space where some treatment groups are completely absent. In these regions, we cannot reliably estimate the causal effect of the treatment, as we have no observed data to inform the counterfactual.

**How to Use It:** Diagnosing positivity violations helps identify regions of the data where causal inference is unreliable. If detected, we must acknowledge the limitations of our causal estimates and may need to either:
*   Adjust the target population to focus on regions with positivity.
*   Collect more data to establish better overlap.
*   Modify the treatment regime of interest, if possible.
*   Consider more advanced methods like causal transport or leveraging external data.
*   Acknowledge and transparently report the limitations of the inference.

**Extrapolation:** Extrapolation involves making predictions or inferences outside the range of the observed data. In causal inference, extrapolation occurs when we attempt to estimate the causal effect of a treatment for covariate values that are not represented in our dataset, or for treatment regimes never observed in the study. If we don't have observations under specific treatment regimes and values of covariates, we are essentially extrapolating the causal effects beyond what we have seen. This extrapolation can be highly sensitive to model misspecification because the model's behavior outside the observed range is unconstrained by the data.

**How to Use It:**  Extrapolation requires caution.  While sometimes necessary, it is crucial to assess the plausibility of the assumed causal relationships beyond the observed data.  Sensitivity analysis and domain expertise become critical. Possible ways to address extrapolation issues:
*   Restrict analysis to areas with empirical support.
*   Use more robust models less sensitive to extrapolation.
*   Explicitly model the extrapolation assumptions and perform sensitivity analysis.
*   Collect additional data to cover the extrapolated region.

## 2) Application scenario

**Scenario:  Treatment for Heart Failure**

Suppose we are studying the effectiveness of a new drug (Treatment *A*) for heart failure.  The treatment assignment depends on patient age (*X1*) and disease severity (*X2*).

*   **Positivity Violation Example:** Imagine that in our dataset, no patients under 50 years old ( *X1* < 50) with severe disease (high *X2*) ever received the new drug (*A*=1).  This could be due to a doctor's protocol or some unknown selection bias. Attempting to estimate the effect of the drug on these patients would violate the positivity assumption as there is no overlap.

*   **Extrapolation Example:**  Our study only includes patients aged 50-80 years old (*X1* is between 50 and 80). We want to estimate the effect of the drug on patients aged 85. This is an extrapolation problem, because we are trying to extend our findings beyond the observed range of age. The effect of the drug might be very different for much older patients due to age-related physiological changes.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np

def check_positivity(df, treatment_col, covariate_cols):
    """
    Checks for positivity violations in a dataset.

    Args:
        df: Pandas DataFrame containing the data.
        treatment_col: Name of the column representing the treatment variable.
        covariate_cols: List of column names representing the covariates.

    Returns:
        A Pandas DataFrame indicating positivity violations.  Each row represents a 
        unique combination of covariate values, and columns show the treatment levels
        and their observed frequencies. Rows where any treatment frequency is zero indicate
        a potential positivity violation.  Returns None if the DataFrame is empty.
    """
    if df.empty:
        return None

    grouped = df.groupby(covariate_cols)[treatment_col].value_counts(normalize=True).unstack(fill_value=0)

    # Identify combinations with zero probability for any treatment
    violation_rows = grouped.isin([0]).any(axis=1)
    violations = grouped[violation_rows]
    violations['positivity_violated'] = True
    return violations

def check_extrapolation(df, covariate_cols, new_data):
  """
  Checks if `new_data` contains values outside the range of `df` for specified covariates.

  Args:
      df: Pandas DataFrame representing the original data.
      covariate_cols: List of column names to check for extrapolation.
      new_data: Pandas DataFrame representing the new data points to be predicted.

  Returns:
      Pandas DataFrame indicating rows in `new_data` that are extrapolations with an extrapolation_flag.
      Returns None if the DataFrames are empty.
  """

  if df.empty or new_data.empty:
        return None

  extrapolation_flags = []
  for index, row in new_data.iterrows():
      extrapolation = False
      for col in covariate_cols:
          min_val = df[col].min()
          max_val = df[col].max()
          if row[col] < min_val or row[col] > max_val:
              extrapolation = True
              break
      extrapolation_flags.append(extrapolation)

  new_data['extrapolation_flag'] = extrapolation_flags
  return new_data[new_data['extrapolation_flag'] == True] #Return only the extrapolation flags
# Example Usage (Simulated Data)
np.random.seed(42) # For reproducibility
n = 100

data = {
    'age': np.random.randint(40, 70, n), # Simulated Age
    'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], n, p=[0.3, 0.4, 0.3]),
    'treatment': np.random.choice([0, 1], n, p=[0.6, 0.4])
}

df = pd.DataFrame(data)

# Positivity Check
covariates = ['age', 'severity']
treatment_col = 'treatment'
positivity_violations = check_positivity(df, treatment_col, covariates)

if positivity_violations is not None and not positivity_violations.empty:
    print("Positivity Violations Detected:\n", positivity_violations)
else:
    print("No Positivity Violations Detected.")


# Extrapolation Check
new_data_points = pd.DataFrame({
    'age': [30, 75],  # Age values outside the range of the original data
    'severity': ['Moderate', 'Severe'],
    'treatment': [0, 1]
})

extrapolations = check_extrapolation(df, covariates, new_data_points)

if extrapolations is not None and not extrapolations.empty:
    print("\nExtrapolations Detected:\n", extrapolations)
else:
    print("No Extrapolations Detected.")
```

**Explanation:**

*   `check_positivity`: This function groups the data by the specified covariates and then calculates the proportion of each treatment level within each group.  If any treatment level has a zero proportion for a particular combination of covariates, it flags a potential positivity violation.

*   `check_extrapolation`:  This function checks if the values in `new_data` for the specified covariates fall within the range of the same covariates in the original data (`df`). If a value is outside the range, it indicates extrapolation.

**Limitations:**

*   The `check_positivity` function only flags *potential* violations. More sophisticated methods might be needed to confirm the severity and impact.
*   The `check_extrapolation` function only checks for violations in the ranges. It does not check for combinations of covariates or density of points in specific regions.
*   For continuous covariates, discretizing might be necessary before applying `check_positivity`, but this comes with its own set of considerations.

## 4) Follow-up question

How can propensity score methods (e.g., inverse probability of treatment weighting) be affected by positivity violations and extrapolation, and what strategies can be used to mitigate these effects when using propensity scores for causal inference?
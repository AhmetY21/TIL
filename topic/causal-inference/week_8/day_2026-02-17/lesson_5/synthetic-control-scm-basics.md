---
title: "Synthetic Control (SCM) Basics"
date: "2026-02-17"
week: 8
lesson: 5
slug: "synthetic-control-scm-basics"
---

# Topic: Synthetic Control (SCM) Basics

## 1) Formal definition (what is it, and how can we use it?)

Synthetic Control Method (SCM) is a statistical method used to estimate the effect of an intervention or treatment in a single treated unit (e.g., a country, a state, a company) when there is no clear-cut control group available. Instead of relying on a single untreated unit as a control, SCM constructs a *synthetic control* – a weighted average of multiple untreated units – that closely resembles the treated unit *before* the intervention.

The key idea is to choose weights for the control units such that the synthetic control mimics the pre-intervention trajectory of the treated unit as closely as possible with respect to relevant covariates and outcome variable. The difference between the observed outcome of the treated unit after the intervention and the predicted outcome based on the synthetic control is then taken as the estimated treatment effect.

Formally, let:

*   *Y<sub>it</sub>*: Outcome for unit *i* at time *t*.
*   *T<sub>0</sub>*: The time of intervention.
*   *D<sub>it</sub>*: Indicator variable, 1 if unit *i* is treated and *t* ≥ *T<sub>0</sub>*, 0 otherwise.
*   *N*: Total number of units.
*   *N<sub>0</sub>*: Number of control units.
*   *N<sub>1</sub>*: Number of treated units (usually 1).
*   *X<sub>i</sub>*: Vector of pre-intervention covariates for unit *i*.
*   *W = (w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>N<sub>0</sub></sub>)*: Vector of weights for the control units, where Σw<sub>i</sub> = 1 and w<sub>i</sub> ≥ 0.

The synthetic control is then:  Σ w<sub>i</sub> * Y<sub>it</sub> for the control units.  The weights *W* are chosen to minimize the difference between *X<sub>1</sub>* (covariates of the treated unit) and Σ w<sub>i</sub> * X<sub>i</sub> (weighted average of control units' covariates).  Often, this is done by minimizing a distance metric like the mean squared error (MSE) between the pre-intervention outcomes of the treated unit and the synthetic control.

The estimated treatment effect at time *t* after the intervention is then:

δ<sub>t</sub> = Y<sub>1t</sub> - Σ w<sub>i</sub> * Y<sub>it</sub> (for control units).

We can use SCM to:

*   Estimate the causal effect of a policy change (e.g., a tobacco control law, a tax increase).
*   Evaluate the impact of a major event (e.g., a terrorist attack, a natural disaster).
*   Analyze the effects of a new technology or innovation.
*   Assess the performance of a company after a major strategic shift.

## 2) Application scenario

Consider the example of California's Proposition 99, a tobacco control program implemented in 1988. We want to estimate the effect of Proposition 99 on per capita cigarette sales in California.  We have data on cigarette sales and other relevant covariates (e.g., income, education levels, pre-existing tobacco control measures) for California and other U.S. states from 1970 to 2000.

Using SCM, we would:

1.  **Define the treated unit:** California.
2.  **Define the control units:** The remaining U.S. states.
3.  **Define the pre-intervention period:** 1970-1987.
4.  **Define the post-intervention period:** 1988-2000.
5.  **Choose relevant covariates:** Income, education, pre-intervention cigarette sales.
6.  **Find the weights *W*:** Determine the weights for each control state that minimize the difference between California's pre-intervention characteristics (covariates and cigarette sales) and the weighted average of the control states' characteristics.  This results in the "synthetic California."
7.  **Calculate the treatment effect:** Compare the actual per capita cigarette sales in California after 1988 with the predicted sales based on the synthetic California.  The difference is the estimated effect of Proposition 99.

## 3) Python method (if possible)

The `SynthControl` package in Python is commonly used for implementing SCM.

```python
import pandas as pd
import numpy as np
from synthcontrol import SynthControl

# Sample data (replace with your actual data)
# Create dummy data for demonstration purposes.
np.random.seed(42)

years = np.arange(1970, 2001)
n_states = 10
california_index = 0  # California is the first state

data = pd.DataFrame(index=years)
for i in range(n_states):
    data[f'state_{i}'] = np.random.randn(len(years)).cumsum() + i*10 # Simulate different trajectories

# Create a covariate (e.g., income)
covariate_data = pd.DataFrame(index = years)
for i in range(n_states):
  covariate_data[f'state_{i}'] = np.random.rand(len(years)) + i*5 # Simulate Income data


# Define intervention year
intervention_year = 1988

# Prepare data for SynthControl
Y = data.values
X = covariate_data.values
treatment_unit = california_index
treated_period = np.where(years >= intervention_year)[0]
control_units = [i for i in range(n_states) if i != california_index]
pre_period = np.where(years < intervention_year)[0]

# Fit the Synthetic Control model

sc = SynthControl(Y, X, treatment_unit, treated_period, control_units, pre_period)
sc.train()

# Get the estimated treatment effect
treatment_effect = sc.effect

print("Estimated Treatment Effect:\n", treatment_effect)


# You can also plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(years, data[f'state_{california_index}'], label='California (Actual)')
plt.plot(years, sc.synthetic_outcome, label='Synthetic California')
plt.axvline(x=intervention_year, color='r', linestyle='--', label='Intervention')
plt.xlabel('Year')
plt.ylabel('Cigarette Sales (Simulated)')
plt.title('Synthetic Control: California Cigarette Sales')
plt.legend()
plt.show()

# Get the weights assigned to the control units
weights = sc.weights
print("\nWeights assigned to control units:")
for i, weight in enumerate(weights):
    print(f"State {control_units[i]}: {weight:.4f}")
```

**Explanation:**

1.  **Import Libraries:** Import `pandas`, `numpy`, and `SynthControl`.
2.  **Create Sample Data:**  This is the most important part. You'll replace this with your actual data. The data needs to be in a `pandas DataFrame` format suitable for `SynthControl`.  Crucially, the data should be organized with time points as rows and units (states) as columns.
3.  **Define Key Parameters:**  Specify the intervention year, the index of the treated unit (California), and which units are considered controls.
4.  **Prepare Data for `SynthControl`:**  The `SynthControl` class takes the data as a NumPy array. It also takes the index of the treated unit, the indices of the treated period (post-intervention), the control units, and the pre-intervention period.
5.  **Fit the Model:** Create a `SynthControl` object and train it using `sc.train()`.  The weights are chosen to minimize the MSE in the pre-intervention period, given the selected covariates.
6.  **Get Treatment Effect:**  `sc.effect` provides the estimated treatment effect. This is a vector of the differences between California's observed outcome and the synthetic control's predicted outcome for each year *after* the intervention.
7.  **Plot the Results (Optional):**  The code includes plotting to visualize the actual outcome for the treated unit and the synthetic control.
8.  **Get Weights:** The weights assigned to each control unit are stored in `sc.weights`. These weights show the relative importance of each control state in constructing the synthetic control.

**Important Notes:**

*   **Data Preparation:** The quality of your data is critical. Ensure your data is clean, complete, and properly formatted.
*   **Covariate Selection:**  Carefully choose covariates that are predictive of the outcome and are not affected by the intervention. This is a crucial step.
*   **Placebo Tests:** Conduct placebo tests (e.g., applying the SCM to other states as if they were treated) to assess the robustness of your results. This can help determine if the observed effect is truly due to the intervention or simply due to chance.
*   **Sensitivity Analysis:**  Explore how the results change with different choices of covariates and pre-intervention periods.
*   **Package Installation:** You may need to install the `SynthControl` package: `pip install synthcontrol`

## 4) Follow-up question

How do I assess the uncertainty or statistical significance of the treatment effects estimated by SCM, given that it's a method designed for single treated units and doesn't readily provide standard errors in the traditional sense? What are some common approaches to addressing this?
---
title: "Practical Diagnostics: Balance, Overlap, Placebos, Falsification"
date: "2026-02-19"
week: 8
lesson: 4
slug: "practical-diagnostics-balance-overlap-placebos-falsification"
---

# Topic: Practical Diagnostics: Balance, Overlap, Placebos, Falsification

## 1) Formal definition (what is it, and how can we use it?)

Causal inference relies on several assumptions to ensure the identified effect is truly causal and not driven by confounding. When these assumptions are violated, our causal estimates can be biased. Practical diagnostics are a suite of techniques used to assess the plausibility of these assumptions in real-world data. The four main diagnostics are:

*   **Balance:**  Balance refers to the similarity of the treatment and control groups *before* the intervention.  Specifically, it concerns whether covariates (observed confounders) are similarly distributed across the treated and untreated groups.  Good balance suggests that the groups were comparable before treatment, lending credibility to the assumption of no unobserved confounding. Poor balance indicates a potential for bias as the treatment assignment might be related to these covariates. We use balance checks to assess the plausibility of the ignorability assumption (no unobserved confounders).  Techniques like comparing means, variances, or distributions of covariates are used.

*   **Overlap (Positivity/Common Support):** Overlap, also called positivity or common support, means that for every set of covariate values, there's a non-zero probability of receiving both the treatment and the control. This means there's some treated and control units sharing the same characteristics. Lack of overlap implies that for some individuals or groups of individuals, there is no counterfactual available; we only observe what happened under treatment *or* control. Without overlap, causal inference relies on extrapolation, which is risky. Overlap is assessed by visualizing propensity scores (probability of treatment given covariates) and ensuring there are substantial regions of common support between the treatment and control groups. Density plots, histograms, and scatter plots of propensity scores are common tools.

*   **Placebos:** Placebo tests involve applying the same causal inference methods but using a *fake* treatment that we know *should* have no causal effect on the outcome. For example, we might assign a random variable as a "treatment" and see if our causal model finds a statistically significant effect on the outcome. If it does, this suggests that the model might be picking up spurious correlations or that there are other forms of bias (e.g. omitted variable bias). This helps check the robustness of our identification strategy. A statistically significant placebo effect casts doubt on the validity of the true treatment effect estimate.

*   **Falsification (Negative Controls):** Similar to placebos, falsification involves testing the effect of the treatment on an outcome that we *a priori* believe *should not* be affected by the treatment (a "negative control outcome"). Alternatively, we could examine the treatment effect at a *prior* time period before the treatment was administered ("pre-treatment outcome").  If we find a statistically significant effect on these negative controls or pre-treatment outcomes, it suggests the presence of confounding or other biases that undermine the validity of our causal estimate. Falsification tests help evaluate the robustness of the model to violations of the causal assumptions.

## 2) Application scenario

Imagine we want to evaluate the effect of a new job training program (treatment) on participants' income (outcome) one year after completing the program.

*   **Balance:** We would check if the participants in the training program (treated group) are similar to those who didn't participate (control group) in terms of age, education level, previous income, skills etc. If the participants in the program are generally more educated than the control group, we have an imbalance on education, which could confound the results.

*   **Overlap:** We would check if there are individuals with similar education and experience levels in both the treatment and control groups. If all individuals with very low education levels are assigned to the control group and never receive the training, there's a lack of overlap in the low-education region of covariate space.  We cannot reliably estimate the effect of the training on those with very low levels of education.

*   **Placebo:** We create a fake "treatment" by randomly assigning individuals to a fake training program.  If this fake program shows a significant impact on income, our causal model might be biased or susceptible to spurious correlations.

*   **Falsification:** We examine the effect of the training program on income *before* the program was implemented. If we see a statistically significant change in pre-treatment income between the treatment and control groups, this suggests that the two groups weren't comparable to begin with or that there's some other factor influencing both the treatment and the past income.  Another falsification test could involve examining the effect of the training program on *unrelated* variables like the participants' likelihood of owning a specific type of pet.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data (replace with your actual data)
np.random.seed(42)
n = 200
data = pd.DataFrame({
    'age': np.random.randint(20, 60, n),
    'education': np.random.randint(8, 16, n),
    'previous_income': np.random.randint(20000, 60000, n),
    'treatment': np.random.choice([0, 1], n, p=[0.6, 0.4]), #treatment assignment
    'outcome': 0 #placeholder
})

# Simulate outcome variable (dependent on treatment and confounders)
data['outcome'] = 5000 + 2000 * data['treatment'] + 100 * data['age'] + 500 * data['education'] + np.random.normal(0, 5000, n)


# Balance Check (using mean differences)
treatment_group = data[data['treatment'] == 1]
control_group = data[data['treatment'] == 0]

balance_table = pd.DataFrame({
    'Treatment Mean': treatment_group[['age', 'education', 'previous_income']].mean(),
    'Control Mean': control_group[['age', 'education', 'previous_income']].mean(),
})
balance_table['Absolute Difference'] = abs(balance_table['Treatment Mean'] - balance_table['Control Mean'])
print("Balance Table:\n", balance_table)



# Overlap Check (using propensity scores)
X = data[['age', 'education', 'previous_income']]
y = data['treatment']
propensity_model = LogisticRegression(random_state=0).fit(X, y)
propensity_scores = propensity_model.predict_proba(X)[:, 1]
data['propensity_score'] = propensity_scores

# Plot propensity score distributions for treatment and control
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='propensity_score', hue='treatment', kde=True)
plt.title('Propensity Score Distribution')
plt.show()


# Placebo Test
data['placebo_treatment'] = np.random.choice([0, 1], n) #Randomly assign a treatment variable

placebo_model = sm.OLS(data['outcome'], sm.add_constant(data['placebo_treatment'])).fit() #linear regression example
print("\nPlacebo Test Results:\n", placebo_model.summary())


# Falsification Test (using pre-treatment outcome - assuming we had it.)
# In this example, we pretend 'previous_income' is the pre-treatment outcome.
falsification_model = sm.OLS(data['previous_income'], sm.add_constant(data['treatment'])).fit()
print("\nFalsification Test Results (on previous income):\n", falsification_model.summary())

```

## 4) Follow-up question

How do you adjust your causal inference strategy when you find violations of balance or overlap assumptions? What are some specific techniques to mitigate these issues?
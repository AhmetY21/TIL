---
title: "Time-Varying Treatments and Time-Varying Confounding"
date: "2026-02-18"
week: 8
lesson: 2
slug: "time-varying-treatments-and-time-varying-confounding"
---

# Topic: Time-Varying Treatments and Time-Varying Confounding

## 1) Formal definition (what is it, and how can we use it?)

Time-varying treatments and time-varying confounding arise in longitudinal studies where both the treatment an individual receives and the confounding variables that affect both treatment and outcome change over time.  Critically, **past treatments can influence future confounders, and these future confounders can, in turn, affect future treatments and the outcome.** This creates a complex feedback loop that standard regression or propensity score matching approaches cannot adequately handle. This is because these methods can lead to bias by inappropriately conditioning on variables that are *intermediate* between treatment and outcome (i.e., are caused by past treatment), thereby blocking causal pathways.

More formally:

*   **Time-Varying Treatment:** A treatment that is administered (or experienced) at multiple time points (t = 1, 2, ..., T) and can vary for each individual and time point. We represent the treatment at time *t* as A<sub>t</sub>.
*   **Time-Varying Confounding:** A confounding variable (or set of variables) that also changes over time and influences both the treatment and outcome at subsequent time points. Importantly, the *value* of these confounders at time *t* can be *affected* by prior treatments. We represent the confounders at time *t* as L<sub>t</sub>.
*   **Outcome:** The outcome of interest, measured at the end of the observation period, usually T. We represent the outcome as Y.

The problem arises because standard methods can mistakenly adjust for variables (L<sub>t</sub>) that lie on the causal pathway from *past* treatments (A<sub>t-1</sub>) to the outcome (Y).  By adjusting for these intermediate variables, we inadvertently block a portion of the total causal effect of the treatment.

**How can we use it?** Recognizing the presence of time-varying confounding requires careful consideration of the causal relationships between treatment, confounders, and outcome *over time*. This understanding then informs the selection of appropriate causal inference methods designed to handle this type of confounding. Methods like g-computation, inverse probability of treatment weighting (IPTW) using marginal structural models, and targeted maximum likelihood estimation (TMLE) are commonly employed. These methods aim to estimate the counterfactual outcome under different treatment regimes by correctly accounting for the dynamic nature of the confounders and the treatments. Essentially, they aim to simulate what would have happened if everyone had followed a specific treatment plan.

## 2) Application scenario

Imagine a study investigating the effect of a specific exercise program (A) on weight loss (Y) over a year (52 weeks, indexed by *t*).

*   **Time-Varying Treatment (A<sub>t</sub>):** Whether or not a participant participates in the exercise program in a given week. This could be binary (yes/no) or a measure of intensity (e.g., hours per week).
*   **Time-Varying Confounding (L<sub>t</sub>):** An individual's diet in a given week, their stress level, their baseline fitness level, and their motivation level.  These factors are all known to influence both participation in the exercise program and weight loss.  Crucially, participation in the exercise program in *previous* weeks (A<sub>t-1</sub>) can *affect* a participant's diet and motivation in *subsequent* weeks (L<sub>t</sub>). For example, someone who regularly exercises might be more motivated to eat healthily (diet and motivation are part of L<sub>t</sub>).
*   **Outcome (Y):** Weight loss at the end of the year.

If we simply regress weight loss on participation in the exercise program, we will likely get a biased estimate of the true effect. This is because we are not accounting for the fact that previous exercise can affect diet and motivation, which in turn affect weight loss. Adjusting for diet and motivation *without* accounting for their dependence on past exercise would block an important causal pathway through which exercise affects weight loss. G-computation, IPTW, or TMLE would be more appropriate here to estimate the effect of different exercise strategies (e.g., exercising every week vs. not exercising at all) on weight loss.

## 3) Python method (if possible)

While there isn't a single "one-size-fits-all" Python function to solve this, libraries like `dowhy`, `statsmodels`, and `linearmodels` can be used in conjunction with custom code to implement methods like g-computation or IPTW for time-varying treatments and confounders.

Here's a basic example illustrating IPTW using `statsmodels` and conceptualized as a function:

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def iptw_estimation(df, treatment_col, confounders_cols):
    """
    Estimates the average treatment effect using inverse probability of treatment weighting (IPTW)
    for time-varying treatments and confounders. This is a simplified conceptual illustration.

    Args:
        df (pd.DataFrame): DataFrame containing treatment, confounders, and outcome data.  Needs columns:
                         'id' (unique identifier for each individual),
                         'time' (time period),
                         treatment_col (treatment variable name),
                         confounders_cols (list of confounder variable names),
                         'outcome' (outcome variable name)
        treatment_col (str): Name of the column representing the treatment variable.
        confounders_cols (list): List of column names representing the confounder variables.

    Returns:
        float: Estimated average treatment effect.  Returns None if the model fails to converge.
    """

    # 1. Estimate propensity scores for each time point
    df['propensity_score'] = None # initialize
    for t in df['time'].unique():
        data_t = df[df['time'] == t]
        try: # needed because propensity scores can be 0 or 1
            formula = f"{treatment_col} ~ " + " + ".join(confounders_cols)
            model = smf.glm(formula=formula, data=data_t, family=sm.families.Binomial()).fit()
            df.loc[df['time'] == t, 'propensity_score'] = model.predict(data_t)
        except:
            print(f"Propensity score model failed to converge for time {t}.  Returning None.")
            return None


    # 2. Calculate weights
    # Note: This is a simplified example and requires handling of 0 or 1 propensity scores.
    # Proper handling often involves trimming or using a stabilized weight.
    df['weight'] = 1.0
    for t in df['time'].unique():
        df.loc[df['time'] == t, 'weight'] = df.loc[df['time'] == t, 'weight'] / df.loc[df['time'] == t, 'propensity_score']

    # 3. Estimate the treatment effect using weighted regression
    try:
        weighted_model = smf.wls(formula=f"outcome ~ {treatment_col}", data=df, weights=df['weight']).fit()
        treatment_effect = weighted_model.params[treatment_col]
        return treatment_effect
    except:
        print("Weighted regression model failed to converge. Returning None.")
        return None

# Example Usage (Illustrative)

# Create some dummy data
data = {'id': [1, 1, 2, 2],
        'time': [1, 2, 1, 2],
        'treatment': [0, 1, 1, 0],
        'confounder1': [1, 2, 3, 4],
        'confounder2': [5, 6, 7, 8],
        'outcome': [2, 4, 6, 8]}
df = pd.DataFrame(data)
df['outcome'] = df['outcome'] + df['treatment'] # Treatment effect

treatment_column = 'treatment'
confounder_columns = ['confounder1', 'confounder2']

ate = iptw_estimation(df.copy(), treatment_column, confounder_columns)  #use copy to not change dataframe in place

if ate is not None:
    print(f"Estimated Average Treatment Effect (ATE): {ate}")
else:
    print("IPTW estimation failed.")

```

**Important Considerations for IPTW (and why this is a simplified illustration):**

*   **Positivity/Overlap:** IPTW requires the *positivity* assumption to hold: that there is a non-zero probability of receiving each treatment value at each time point, *conditional on the confounders*. If propensity scores are too close to 0 or 1, the weights become extremely large and the variance of the estimate explodes. Solutions include trimming weights or using stabilized weights.
*   **Stabilized Weights:** Stabilized weights can improve the stability of the IPTW estimator. They involve estimating a numerator for the weight that reflects the marginal probability of treatment.
*   **Model Misspecification:** The IPTW estimator is sensitive to misspecification of the propensity score model. Diagnostics and sensitivity analyses are crucial.
*   **Zero Propensity Scores:** If any individual has a propensity score of exactly 0, the weight becomes infinite.  Solutions include trimming (setting a maximum weight) or using stabilized weights with truncation.

This code provides a starting point for understanding how IPTW can be implemented. It is highly recommended to consult more comprehensive resources and libraries dedicated to causal inference for robust and reliable analyses. Libraries like `causalml` might also provide helpful tools. The `statsmodels` library is used here to perform the regression tasks that are the core of the algorithm. The general approach is to first use logistic regression (`smf.glm`) to calculate the propensity scores, then an ordinary least squares regression (`smf.wls`) weighted by the previously calculated propensity scores to get an estimate of the ATE.

## 4) Follow-up question

Given that g-computation, IPTW, and TMLE are all methods for handling time-varying treatments and confounders, what are the key advantages and disadvantages of each, and how would you choose the most appropriate method for a given research question and dataset?
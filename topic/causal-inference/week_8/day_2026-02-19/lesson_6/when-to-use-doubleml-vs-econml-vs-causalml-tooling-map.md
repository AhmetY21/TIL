---
title: "When to Use DoubleML vs EconML vs CausalML (Tooling Map)"
date: "2026-02-19"
week: 8
lesson: 6
slug: "when-to-use-doubleml-vs-econml-vs-causalml-tooling-map"
---

# Topic: When to Use DoubleML vs EconML vs CausalML (Tooling Map)

## 1) Formal definition (what is it, and how can we use it?)

This topic focuses on guiding users to select the appropriate Python package (DoubleML, EconML, or CausalML) for their causal inference problem. Each package offers distinct strengths and caters to different causal inference tasks, estimators, and computational needs. Selecting the right tool is crucial for efficient and reliable causal inference. These packages provide implementations of cutting-edge methods that allow researchers and practitioners to estimate causal effects while addressing confounding.

*   **DoubleML:** Primarily focuses on Double Machine Learning (DML) methods. DML leverages machine learning algorithms to control for confounding and estimate treatment effects. It's particularly useful when dealing with high-dimensional data where traditional parametric methods might struggle. Key advantage is its robust bias-correction through cross-fitting and orthogonality conditions. It is well suited when you want to estimate treatment effects controlling for confounders and have flexible estimators for the nuisance functions.

*   **EconML:** A broader package offering a variety of causal machine learning estimators, encompassing but not limited to DML.  It emphasizes flexibility and integrates well with scikit-learn.  EconML includes various estimators based on heterogeneous treatment effects, such as meta-learners, tree-based methods, and orthogonal random forests. Its core purpose is to estimate heterogeneous treatment effects, understanding how treatment effects vary across different subpopulations.

*   **CausalML:** Focuses heavily on Uplift Modeling and Causal Machine Learning for customer relationship management (CRM) and marketing applications. It provides a comprehensive suite of algorithms designed to estimate the incremental impact of a treatment (e.g., marketing campaign) on individual customers.  The primary use is in predicting individual treatment effects to optimize targeting decisions. It's designed to directly optimize decisions based on causal inference results.

Essentially, the tooling map helps navigate these packages based on problem type, estimator needed, and specific features (e.g., uplift modeling).

## 2) Application scenario

Consider these examples:

*   **Scenario 1 (High-Dimensional Confounding):** Estimating the effect of a new job training program on future earnings, controlling for a large number of demographic and socioeconomic factors. DoubleML would be a good choice due to its strength in handling high-dimensional data.

*   **Scenario 2 (Heterogeneous Treatment Effects):** Determining how the effect of a drug on blood pressure varies across different patient subgroups based on age, gender, and pre-existing conditions. EconML would be suitable as it offers diverse estimators for heterogeneous treatment effects.

*   **Scenario 3 (Uplift Modeling for Marketing):** Identifying customers who are most likely to purchase a product *because* of a marketing campaign, rather than purchasing it anyway. CausalML is designed for this specific scenario.

*   **Scenario 4 (Instrumental Variables):** Estimating the effect of years of education on earnings, using distance to college as an instrument. EconML offers estimators that incorporate instrumental variables. DoubleML can also incorporate IVs.

## 3) Python method (if possible)

Here's a simplified illustration using Python (with placeholders as full code requires data):

```python
# Using DoubleML for estimating Average Treatment Effect (ATE)
import doubleml as dml
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd

# Assume you have X, d, y (features, treatment, outcome)

# Create dummy data for demonstration
np.random.seed(42)
n_obs = 100
X = pd.DataFrame(np.random.rand(n_obs, 5), columns=['X1', 'X2', 'X3', 'X4', 'X5'])
d = pd.Series(np.random.randint(0, 2, n_obs))
y = pd.Series(np.random.rand(n_obs))


# Define ML models
ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)  #Outcome model
ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42) #Treatment model

#Initialize doubleml data
dml_data = dml.DoubleMLData(X, y, d)

# Initialize DoubleMLIRM object
dml_irm_obj = dml.DoubleMLIRM(dml_data, ml_l, ml_m) #IRM: Interactive Regression Model (ATE focused)
dml_irm_obj.fit()

print("ATE estimate (DoubleML):", dml_irm_obj.ate)


# Using EconML for heterogeneous treatment effects with metalearners
from econml.metalearners import TLearner, XLearner, SLearner
from sklearn.linear_model import LinearRegression

# Initialize metalearner
tl = TLearner(models=LinearRegression()) #can use more complex models

# Fit metalearner
tl.fit(Y=y, T=d, X=X)

# Estimate ITE
ite_predictions = tl.effect(X)
print("Example ITE (EconML):", ite_predictions[:5]) # First 5 ITE predictions


# Using CausalML for Uplift Modeling

#This would require more data preparation tailored for uplift modeling.
#Here's just the general outline:
from causalml.uplift import LogisticRegressionUplift, MetalearnerUplift

# Initialize uplift model
up = MetalearnerUplift(learner=LogisticRegressionUplift()) #Can swap the learner
# fit the model to training data
#up.fit(X_train, treatment, y_train) #Placeholder - requires uplift data
print ("CausalML Requires data suitable for uplift modeling.") # placeholder code
```

**Explanation:**

*   **DoubleML:** Demonstrates estimating the Average Treatment Effect (ATE) using DoubleML. We create `DoubleMLData`, specify machine learning models for the outcome and treatment assignments, initialize a `DoubleML` object (IRM, focusing on ATE), fit the model, and print the ATE estimate.
*   **EconML:** Shows how to use `TLearner` (a metalearner) to estimate individual treatment effects (ITE). We initialize a `TLearner` with a linear regression model, fit the model to the data, and then use the `effect` method to predict ITEs for each individual.
*   **CausalML:** Provides a snippet showcasing the basic structure of using `CausalML` for uplift modeling.  The data preparation is significantly different for uplift modeling, involving treatment and control groups and focus on the *incremental* effect of treatment. This example is incomplete because creating suitable data is beyond the scope of this brief demonstration. `CausalML` offers specifically designed methods for optimizing marketing campaigns, not just prediction.

**Tooling Map Summary:**

| Feature          | DoubleML                  | EconML                     | CausalML                       |
|------------------|---------------------------|----------------------------|-------------------------------|
| Primary Focus    | Bias-corrected ATE/ATE Estimation | Heterogeneous Treatment Effects | Uplift Modeling/CRM         |
| Data Dimensions  | Handles High-Dimensional | Flexible, works well with scikit-learn | Assumes CRM/Marketing data  |
| Estimators       | DML-based                | Wide range of meta-learners, forest-based, IV | Uplift-specific learners     |
| Key Advantage    | Robust Bias Correction   | Flexibility, Integration   | CRM/Marketing Optimization    |
| Use Case         | Causal Effects w/ Confounding | Understanding Subgroup Effects  | Targeted Marketing Campaigns |

## 4) Follow-up question

How do I deal with model selection within each of these packages, especially when choosing the right machine learning model for nuisance functions in DoubleML or the appropriate meta-learner in EconML? How do the model assumptions differ across these packages and their estimators?
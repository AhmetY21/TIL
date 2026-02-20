---
title: "When to Use DoubleML vs EconML vs CausalML (Tooling Map)"
date: "2026-02-20"
week: 8
lesson: 1
slug: "when-to-use-doubleml-vs-econml-vs-causalml-tooling-map"
---

# Topic: When to Use DoubleML vs EconML vs CausalML (Tooling Map)

## 1) Formal definition (what is it, and how can we use it?)

This topic deals with understanding the appropriate use cases for three popular Python libraries used in causal inference: DoubleML, EconML, and CausalML. It's essentially a tooling map that helps you choose the right tool for a specific causal inference problem based on the problem's characteristics, such as the complexity of the causal effect you are trying to estimate (ATE, CATE, IV, etc.), the presence of high-dimensional covariates, the types of machine learning algorithms you want to use, and the desired level of interpretability and diagnostics.  Choosing the correct tool allows for more efficient and accurate causal effect estimation.

*   **DoubleML (Double/Debiased Machine Learning):** Focuses on estimating average treatment effects (ATE) and heterogeneous treatment effects (CATE) using a double machine learning approach. DoubleML leverages machine learning for nuisance functions (e.g., propensity score, outcome regression) and then uses a debiased estimator to obtain consistent and asymptotically normal estimates of the causal effect, even when the machine learning models are only consistent at a slower rate than the parametric rate. It shines when you have complex, potentially nonlinear relationships between covariates and the outcome and treatment variables, but you're primarily interested in the average effect or conditional average effect. A key advantage is its theoretical guarantees and robustness to model misspecification.

*   **EconML (Econometric Machine Learning):** A broader library offering a rich collection of estimators for various causal inference problems. EconML provides implementations of many state-of-the-art causal inference techniques, including instrumental variables estimation, policy learning, and heterogeneous treatment effect estimation. EconML excels in settings with complex causal structures and supports a wide variety of machine learning models for estimating nuisance functions. It's particularly well-suited for situations where you need to estimate treatment effects that vary based on individual characteristics, understand the causal effect of multiple treatments, or handle confounding through instrumental variables.

*   **CausalML (Causal Machine Learning):** Primarily concerned with estimating heterogeneous treatment effects (CATE) using a variety of machine learning methods, focusing on uplift modeling and targeting. CausalML's focus is more on predicting individual treatment effects for optimization purposes, for instance, deciding which customers to target with a specific marketing campaign to maximize the response rate. The main goal is often to find subpopulations most receptive to an intervention.

In summary:

*   **DoubleML:** Robust ATE/CATE estimation with theoretical guarantees, good for high-dimensional data, emphasizes robustness to model misspecification.
*   **EconML:**  Versatile library for many causal inference problems (ATE, CATE, IV, policy learning), supports various ML models and handles complex causal structures.
*   **CausalML:**  Focus on predicting heterogeneous treatment effects (uplift modeling) for targeting interventions and optimizing treatment assignment.

## 2) Application scenario

Here are some application scenarios where each library would be most appropriate:

*   **DoubleML:** Imagine you are analyzing the effect of a job training program on employment. You have a large dataset with many individual characteristics, and you want to estimate the average impact of the program. You are also concerned about potential confounding and want a robust estimate even if your models for the propensity score or outcome regression are not perfectly specified. DoubleML would be a good choice due to its robust estimation of the ATE, even under model misspecification.

*   **EconML:** Suppose you want to understand the effect of a new drug on patient health outcomes. You suspect that the drug's effectiveness might vary depending on the patient's age, gender, and pre-existing conditions. Furthermore, you have an instrumental variable (e.g., a doctor's recommendation) that affects drug uptake but is otherwise unrelated to patient health. EconML's flexible heterogeneous treatment effect estimators and instrumental variable estimators would be ideal here. You can explore how the treatment effect varies based on patient characteristics and account for endogeneity via the instrumental variable.

*   **CausalML:** Consider a marketing campaign where you want to determine which customers should receive a promotional offer. You have data on customer demographics, past purchase behavior, and responses to previous campaigns. You want to identify the customers most likely to increase their spending if they receive the offer. CausalML's uplift modeling techniques would be well-suited to predict the individual-level treatment effect and target the offer to those most likely to be persuaded.

## 3) Python method (if possible)

Here are simple examples of each library's usage:

**DoubleML:**

```python
from doubleml.datasets import make_pliv_multiplicative_heterogeneity
from doubleml import DoubleMLIVM

# Generate some data (replace with your actual data)
np.random.seed(42)
data = make_pliv_multiplicative_heterogeneity(n_obs=100, dim_z=1, return_type='DataFrame')
X_cols = data.columns[data.columns.str.startswith('X')].tolist()
Z_cols = data.columns[data.columns.str.startswith('Z')].tolist()

# Initialize and fit the DoubleMLIVM model
dml_ivm_obj = DoubleMLIVM(data, 'y', 'd', X_cols, Z_cols)
dml_ivm_obj.fit()

# Access the results
print(dml_ivm_obj.summary)
```

**EconML:**

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import DML

# Generate some data (replace with your actual data)
X = np.random.rand(100, 5)  # Covariates
T = np.random.randint(0, 2, size=100)  # Treatment
Y = np.random.rand(100)  # Outcome

# Initialize and fit the DML model
dml_model = DML(model_y=RandomForestRegressor(),
                  model_t=RandomForestRegressor(),
                  random_state=42)
dml_model.fit(Y, T, X=X)

# Estimate the average treatment effect
ate = dml_model.ate(X)
print("ATE:", ate)
```

**CausalML:**

```python
from causalml.inference.tree import UpliftTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = pd.DataFrame({'feature_1': np.random.rand(100),
                     'feature_2': np.random.rand(100),
                     'treatment': np.random.randint(0, 2, size=100),
                     'outcome': np.random.rand(100)})

X = data[['feature_1', 'feature_2']]
y = data['outcome']
treatment = data['treatment']

# Split the data
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X, y, treatment, test_size=0.2, random_state=42)

# Train the UpliftTreeClassifier
uplift_model = UpliftTreeClassifier(max_depth=5, min_samples_leaf=10, min_samples_treatment=5,
                                   min_samples_control=5)
uplift_model.fit(X_train.values, treat_train.values, y_train.values)


# Predict uplift scores
uplift = uplift_model.predict(X_test.values)
print("Uplift scores:", uplift)
```

**Important Notes:**

*   These are simplified examples for demonstration purposes. Real-world applications require careful data preprocessing, feature engineering, and model tuning.
*   The choice of machine learning models within each library (e.g., RandomForestRegressor, GradientBoostingRegressor, etc.) depends on the specific data and the problem at hand.
*   Proper evaluation metrics should be used to assess the performance of the causal inference models.
*   Ensure data meets the assumptions of each method. For example, DoubleML relies on a specific double robustness property.

## 4) Follow-up question

If you're unsure which library to use, what are the first three questions you should ask yourself about your problem?
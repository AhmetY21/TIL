---
title: "Explainability in NLP (LIME, SHAP)"
date: "2026-02-28"
week: 9
lesson: 4
slug: "explainability-in-nlp-lime-shap"
---

# Topic: Explainability in NLP (LIME, SHAP)

## 1) Formal definition (what is it, and how can we use it?)

Explainability in NLP refers to techniques that help us understand *why* a particular NLP model made a specific prediction. Instead of treating models as "black boxes," these methods aim to shed light on the factors that contribute most significantly to the model's output. This understanding allows us to:

*   **Debug models:** Identify biases, flaws, or unexpected behaviors in the model. For example, discover that a sentiment analysis model is unfairly biased against a specific demographic group because of certain keywords.
*   **Build trust:** Increase confidence in the model's predictions by providing a rationale behind them. This is especially important in sensitive applications like healthcare or finance.
*   **Improve model design:** Gain insights into which features the model is actually learning, which can inform feature engineering and model selection.
*   **Ensure fairness and transparency:** Detect and mitigate biases that could lead to discriminatory outcomes.

**LIME (Local Interpretable Model-agnostic Explanations):** LIME explains the predictions of *any* classifier in an interpretable and local way. It approximates the model locally by learning a simple, interpretable model (like a linear model) around the specific prediction we want to explain.  This is done by perturbing the input data slightly, obtaining predictions from the original model for these perturbed samples, and then weighting the perturbed samples by their proximity to the original input. The learned, interpretable model highlights the features that are most important for the prediction in the vicinity of the input.

**SHAP (SHapley Additive exPlanations):** SHAP uses concepts from game theory, specifically Shapley values, to explain the output of *any* machine learning model. Shapley values quantify the contribution of each feature to the prediction.  It considers all possible feature combinations and calculates the average marginal contribution of each feature across all combinations. This provides a more global and consistent explanation compared to LIME, though it can be computationally more expensive. SHAP provides a unified framework that includes several existing explanation methods (e.g., LIME can be considered a special case of SHAP under certain conditions).  SHAP provides individual explanations as well as global feature importances.

## 2) Application scenario

**Sentiment Analysis:** Imagine a model predicting the sentiment (positive, negative, neutral) of movie reviews.

*   **LIME:** We can use LIME to understand *why* the model predicted a specific review as negative. LIME might highlight specific words like "awful," "terrible," or "boring" as the main drivers of the negative prediction for that particular review.
*   **SHAP:** We can use SHAP to understand the global influence of different words on the model's sentiment predictions. SHAP might reveal that the word "amazing" consistently contributes positively to the sentiment score across many reviews, while the word "disappointing" consistently contributes negatively. It can also show how different words interact, for instance, "not bad" can show up as positive.

**Spam Detection:** Consider a model classifying emails as spam or not spam.

*   **LIME:** LIME can identify specific phrases or words (e.g., "urgent," "free offer," "limited time") that led the model to classify a particular email as spam.
*   **SHAP:** SHAP can reveal which features (e.g., sender's domain, number of links, presence of certain keywords) are most important for identifying spam emails across the entire dataset.

## 3) Python method (if possible)

```python
import shap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Sample data (replace with your actual text data)
data = {'text': ["This is a great movie!", "I hated this film", "The acting was superb", "It was a terrible experience", "I loved it!"],
        'sentiment': [1, 0, 1, 0, 1]} # 1 for positive, 0 for negative
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Create a pipeline
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='liblinear'))

# Train the model
pipeline.fit(X_train, y_train)

# Initialize JavaScript visualization for SHAP (only needed once)
shap.initjs()

# Create explainer (SHAP)
explainer = shap.Explainer(pipeline.predict_proba, pipeline[0].transform(X_train)) #pipeline[0] is the TF-IDF vectorizer; pipeline.predict_proba is the predict probability function

# Explain the prediction for a specific example (from the test set)
shap_values = explainer(pipeline[0].transform([X_test.iloc[0]])) # X_test.iloc[0] grabs the first element in the test set

# Visualize the explanation for the first example
shap.force_plot(explainer.expected_value[1], shap_values[0].values, feature_names=pipeline[0].get_feature_names_out(),show=False,matplotlib=True) #showing the shap explanation of the first element in the test set using a force plot

import matplotlib.pyplot as plt
plt.show()
# Global feature importance
shap_values_train = explainer(pipeline[0].transform(X_train))
shap.summary_plot(shap_values_train, feature_names=pipeline[0].get_feature_names_out())
```

**Explanation:**

1.  **Data Preparation:** Create sample text data and split into training and testing sets.
2.  **Model Training:** Create a pipeline with TF-IDF vectorization and Logistic Regression.  Train the pipeline on the training data.
3.  **SHAP Explainer:** Create a `shap.Explainer` object. This requires the model's prediction function (in this case, `pipeline.predict_proba`) and the training data (or a representative subset) used by `shap` to estimate background distributions needed to compute the Shapley values.  The transform function of TF-IDF vectorizer needs to be passed to the explainer to transform the input text into a format that the SHAP explainer expects.
4.  **SHAP Values Calculation:** Calculate SHAP values for a specific example from the test set using the trained explainer.
5.  **Visualization:** Use `shap.force_plot` to visualize the contribution of each word to the prediction.  The `matplotlib=True` argument is added so that it can show the visualization on VSCode. Also included `plt.show()` at the end so the plot will be displayed.
6.  **Global Feature Importance:** Calculate SHAP values on the training dataset. Use `shap.summary_plot` to display the global importance of each feature.

**Note:**  LIME also has Python libraries available (e.g., `lime`). The general process is similar: you create an explainer object, specify the prediction function and the data, and then generate an explanation for a specific instance. The code for LIME would depend on whether it is text classification or regression. The provided example uses SHAP because it is more popular. Also, the sample data is toy data so the results will not be very insightful. Real-world examples will provide more meaningful features. Finally, remember to install `shap`, `sklearn`, `pandas` and `matplotlib`.

## 4) Follow-up question

How can we use explainability techniques to mitigate biases in NLP models that are identified through the explanations generated by LIME or SHAP? For example, if LIME reveals that a model is unfairly relying on gendered pronouns in a job description to predict whether someone is a suitable candidate, what steps can we take?
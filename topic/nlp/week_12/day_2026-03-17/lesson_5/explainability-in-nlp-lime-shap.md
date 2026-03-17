---
title: "Explainability in NLP (LIME, SHAP)"
date: "2026-03-17"
week: 12
lesson: 5
slug: "explainability-in-nlp-lime-shap"
---

# Topic: Explainability in NLP (LIME, SHAP)

## 1) Formal definition (what is it, and how can we use it?)

Explainability in NLP refers to the ability to understand *why* a natural language processing model makes a specific prediction. It aims to shed light on the model's internal decision-making process, identifying the features (e.g., words, phrases, or input features) that are most influential in determining the model's output. This is crucial because:

*   **Debugging:** It helps identify biases, errors, and unexpected behaviors in the model.
*   **Trust:** It increases user trust in the model by providing a rationale for its predictions. If a user understands why a model made a certain prediction, they are more likely to accept and rely on it.
*   **Improvement:** It can provide insights for model improvement by revealing which features are truly important and which are noise.
*   **Compliance:** In regulated industries (e.g., finance, healthcare), explainability is often required to ensure fairness and transparency.

**LIME (Local Interpretable Model-agnostic Explanations)** is a technique that approximates the behavior of a complex model locally with a simpler, interpretable model. For NLP, it highlights the words in a text that contribute most positively or negatively to a model's prediction for that specific instance. LIME works by:

1.  Perturbing the input data (e.g., by removing or adding words in a text).
2.  Obtaining predictions from the black-box model on these perturbed data points.
3.  Weighting the perturbed data points based on their proximity to the original instance.
4.  Training a simpler, interpretable model (e.g., a linear model) on the perturbed data and their corresponding predictions.
5.  Using the coefficients of the simpler model to explain the importance of each feature for the original instance.

**SHAP (SHapley Additive exPlanations)** is a technique based on game theory to explain the output of any machine learning model. It assigns each feature an importance value for a particular prediction. SHAP values represent the average marginal contribution of a feature across all possible coalitions of features. In the context of NLP, SHAP can highlight the importance of different words or phrases in a text for a specific prediction. Key aspects of SHAP include:

*   **Consistency:** If a feature has a larger marginal contribution to the prediction, it will have a larger SHAP value.
*   **Local Accuracy:** The sum of the SHAP values for all features equals the difference between the model's prediction and the average prediction over the entire dataset.
*   **Missingness:** Features that are absent have a SHAP value of 0.

## 2) Application scenario

**Scenario:** Sentiment analysis of customer reviews.

Let's say we have a model that predicts whether a customer review is positive or negative. We want to understand why the model classified a specific review as negative.

**LIME:** We can use LIME to highlight the words that contributed most to the negative sentiment score. For example, LIME might highlight words like "terrible", "disappointed", and "slow" in a review.  This helps us understand if the model is correctly identifying negative sentiment or if it's being influenced by irrelevant words. If the model incorrectly identified a review as negative because of a neutral word used in a negative context, LIME would help expose this.

**SHAP:** We can use SHAP to understand the contribution of each word in the review to the final sentiment score.  SHAP would provide a numerical value for each word, indicating how much that word pushed the prediction towards positive or negative sentiment. This provides a more granular and comprehensive understanding of the model's decision-making process compared to LIME.  We can see which words are contributing the most, and whether the model is relying on specific keywords or a broader understanding of the text.

## 3) Python method (if possible)

Here's an example using the `lime` and `shap` Python libraries.  Note that this is a simplified example and assumes you have a trained sentiment analysis model (`model`) that takes text as input and returns a probability of positive sentiment.

```python
import lime
import lime.lime_text
import shap
import transformers
import numpy as np

# Assume you have a trained model (e.g., using Transformers)
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_proba(texts):
    """
    A wrapper to feed the input to the model correctly and return probabilities.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probas = transformers.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()
    return probas

# Example review
review = "This movie was terrible. The acting was awful and the plot was boring."

# LIME Example
explainer = lime.lime_text.LimeTextExplainer(class_names=['negative', 'positive'])
explanation = explainer.explain_instance(review, predict_proba, num_features=6)
print("LIME Explanation:")
print(explanation.as_list())


# SHAP Example
# Create a SHAP explainer using KernelExplainer (model agnostic but slow).  Faster options
# exist for specific model types (e.g., TransformersExplainer)
explainer = shap.KernelExplainer(predict_proba, shap.sample(np.array([review]),100))
shap_values = explainer.shap_values(review)

print("\nSHAP Explanation:")
# Sum shap values for each word, averaging across classes if necessary
if isinstance(shap_values, list): #multiple output classes
  class_0_shap_values = shap_values[0]
  class_1_shap_values = shap_values[1]
  word_importances = np.mean([class_0_shap_values, class_1_shap_values], axis=0)
else: #single output class
  word_importances = shap_values

words = review.split()
for i, word in enumerate(words):
  print(f"Word: {word}, SHAP value: {word_importances[i]}")


```

**Important Notes:**

*   **Model-Agnostic:** LIME and SHAP (especially KernelSHAP) are model-agnostic, meaning they can be used with any machine learning model. However, some SHAP explainers are specifically designed for certain model types (e.g., `TreeExplainer` for tree-based models, `DeepExplainer` for deep neural networks, `TransformersExplainer` for transformers) and provide faster and more accurate explanations for those models.
*   **Computational Cost:** SHAP can be computationally expensive, especially for large datasets or complex models.
*   **Approximation:** Both LIME and SHAP provide approximations of feature importance. The accuracy of the explanations depends on the parameters used and the complexity of the model.
*   **TransformersExplainer:** For transformer-based models, using `shap.Explainer(model, tokenizer)` or `shap.TransformersExplainer(model, tokenizer)` can offer significantly faster and often more accurate results compared to KernelSHAP. They leverage the internal workings of the Transformer architecture.
*   **SHAP for Transformers:**  With Transformers, it's important to account for the tokenizer. The tokenizer may split words into subwords (e.g., "terrible" might be split into "ter" "##rible"). The SHAP values will be assigned to these subwords. You'll typically need to aggregate these subword SHAP values to get word-level importances.

## 4) Follow-up question

How do the explanations from LIME and SHAP differ in terms of the scope of explanation (local vs. global) and the type of information they provide? Are there situations where one method is preferred over the other? Why?
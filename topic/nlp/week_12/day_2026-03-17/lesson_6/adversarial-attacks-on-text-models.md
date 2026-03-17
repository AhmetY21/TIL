---
title: "Adversarial Attacks on Text Models"
date: "2026-03-17"
week: 12
lesson: 6
slug: "adversarial-attacks-on-text-models"
---

# Topic: Adversarial Attacks on Text Models

## 1) Formal definition (what is it, and how can we use it?)

Adversarial attacks on text models involve crafting small, often imperceptible, perturbations to input text that cause the model to produce incorrect or unintended outputs.  These attacks exploit vulnerabilities in the model's decision boundaries, leading to misclassification, incorrect sentiment analysis, faulty question answering, or other undesirable behaviors.

**Formal Definition:** Let `f(x)` be a text model (e.g., sentiment classifier), where `x` is the input text. An adversarial attack aims to find a perturbed input `x'` such that `d(x, x') ≤ ε`, where `d` is a distance metric (e.g., character-level edit distance, semantic similarity) and `ε` is a predefined threshold, while also ensuring that `f(x) ≠ f(x')` or, more generally, that `f(x')` produces an incorrect or undesired output according to a specific attack goal.  The attacker seeks to minimize the distance `d(x, x')` subject to the constraint that the model's output is changed in the targeted way.

**How can we use it?**

*   **Evaluating Model Robustness:** Adversarial attacks are primarily used to assess the vulnerability of text models to minor input changes.  A model that is easily fooled by adversarial examples is considered less robust.
*   **Adversarial Training:**  By training models on adversarial examples in addition to clean data, we can improve their robustness. This involves generating adversarial examples during training and including them in the training dataset.
*   **Understanding Model Weaknesses:** Analyzing the types of perturbations that are successful in attacking a model can provide insights into the model's weaknesses and biases.
*   **Developing Defense Mechanisms:**  Understanding how attacks work allows researchers to develop defense mechanisms that can detect and mitigate adversarial examples. These mechanisms could include input sanitization, adversarial example detection, or robust model architectures.
*   **Security Auditing:** In real-world applications, adversarial attacks can be used to audit the security of NLP systems, ensuring they are not susceptible to malicious manipulation.

## 2) Application scenario

Consider a sentiment analysis model used to monitor social media for customer feedback about a product. An attacker could craft adversarial examples that appear positive but subtly convey negative sentiments, tricking the model into misclassifying the feedback.

**Example:**

*   **Original Sentence (Negative):** "The product is terrible. It broke after only a week."
*   **Sentiment Analysis Output:** Negative
*   **Adversarial Sentence (Perturbed):** "The product is *awesome*, it broke after only a week *but it's still pretty neat*."  (words in italics are the perturbations)
*   **Sentiment Analysis Output:** Positive (incorrect)

In this scenario, the attacker subtly inserts seemingly positive words like "awesome" and "pretty neat" to mislead the sentiment analysis model. This could damage the accuracy of the company's customer feedback analysis, leading to misinformed decisions.

Another application scenario could be attacking a hate speech detection model.  An attacker could carefully craft hateful messages that evade detection by the model, allowing them to spread offensive content without being flagged.  Similarly, in question answering, attackers could subtly alter questions to elicit incorrect or biased answers from the model.

## 3) Python method (if possible)

The `textattack` library is a popular Python framework for generating adversarial examples for text models.  Here's a simple example demonstrating how to attack a sentiment analysis model using a character-level perturbation attack:

```python
import textattack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load a model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # A DistilBERT model fine-tuned for SST-2 sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Create a TextAttack model wrapper
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# 3. Load a dataset (e.g., the SST-2 validation split)
dataset = HuggingFaceDataset("glue", "sst2", "validation", shuffle=True)

# 4. Choose an attack recipe (e.g., TextFooler)
attack = TextFoolerJin2019.build(model_wrapper)

# 5. Create an attacker
attacker = textattack.Attack(attack, dataset)

# 6. Perform the attack (e.g., on the first 5 examples)
results = attacker.attack_dataset(dataset, indices=range(5))

# Print the results (optional)
for result in results:
    print(result)
```

**Explanation:**

1.  **Load Model and Tokenizer:**  We load a pre-trained DistilBERT model fine-tuned for sentiment analysis (SST-2 dataset) and its corresponding tokenizer.
2.  **Model Wrapper:**  We wrap the Hugging Face model with `HuggingFaceModelWrapper` to make it compatible with TextAttack.
3.  **Load Dataset:**  We load a small subset of the SST-2 validation dataset using `HuggingFaceDataset`.
4.  **Choose Attack Recipe:**  We select the `TextFoolerJin2019` attack, which is a character-level perturbation attack that aims to fool the model by making minimal changes to the input text.
5.  **Create Attacker:**  We create a `textattack.Attack` object that combines the attack recipe and the dataset.
6.  **Perform Attack:**  We use `attacker.attack_dataset()` to generate adversarial examples for the first 5 examples in the dataset. The `indices` parameter specifies which examples to attack. The results are stored in the `results` list.  Each `result` contains the original text, the adversarial example, and the predicted labels by the model for both.

**Installation:**

You'll need to install the necessary libraries:

```bash
pip install textattack transformers datasets
```

This example provides a basic demonstration. TextAttack offers a wide range of attack recipes, defense methods, and evaluation tools for exploring adversarial attacks on text models.

## 4) Follow-up question

What are some common defense mechanisms against adversarial attacks on text models, and how effective are they in practice?
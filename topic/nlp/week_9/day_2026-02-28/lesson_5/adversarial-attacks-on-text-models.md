---
title: "Adversarial Attacks on Text Models"
date: "2026-02-28"
week: 9
lesson: 5
slug: "adversarial-attacks-on-text-models"
---

# Topic: Adversarial Attacks on Text Models

## 1) Formal definition (what is it, and how can we use it?)

Adversarial attacks on text models involve crafting subtly perturbed inputs designed to fool machine learning models into making incorrect predictions. Unlike attacks on image models where perturbations can be almost imperceptible to the human eye, adversarial attacks on text typically involve modifying words, characters, or the sequence of words in a text input. The goal is to find minimal changes that cause the model to misclassify the text, without significantly altering its meaning to a human reader.

Formally, let *x* be the original input text, *f(x)* be the model's prediction for *x*, and *x'* be the adversarial example. The attack aims to find an *x'* such that:

*   *x'* is semantically similar to *x* (i.e., a human would perceive them as having essentially the same meaning). This can be enforced through various constraints like minimum change in edit distance, cosine similarity of word embeddings before/after modification, etc.
*   *f(x')* != *f(x)*, meaning the model's prediction for the adversarial example is different from its prediction for the original input. Ideally, we want *f(x')* to be a targeted misclassification toward a specific incorrect class.
*   The perturbation ||x' - x|| is minimized, i.e., the change between the original and adversarial example is minimal according to a defined metric (e.g., number of words changed, character edit distance).

We can use adversarial attacks to:

*   **Evaluate the robustness of text models:** By testing how easily a model can be fooled, we can assess its vulnerability to adversarial examples. This helps identify weaknesses and areas for improvement.
*   **Train more robust models:** Adversarial training involves training a model on a dataset augmented with adversarial examples. This can improve the model's ability to generalize to unseen and slightly perturbed inputs.  This technique helps make the model more resilient to adversarial perturbations during deployment.
*   **Understand model behavior:** Examining the changes required to fool a model can provide insights into its decision-making process and identify potential biases or vulnerabilities.
*   **Data augmentation:** Use generated adversarial examples to increase the diversity of the training dataset.

## 2) Application scenario

Consider a sentiment analysis system used to monitor customer feedback on social media. A malicious actor wants to manipulate the system to believe negative feedback is positive.  They could craft adversarial examples like this:

Original negative review: "This product is terrible. It broke after only one use." (Model predicts: Negative sentiment)

Adversarial example: "This product is terrible, but only if you want it to last a very, very long time. I guess it didn't break after only one use." (Model might predict: Positive sentiment, due to the added "but" clause and negated breaking)

Here, the attacker subtly altered the wording to reverse the sentiment while maintaining (debatably) plausible English. The application scenario demonstrates how adversarial attacks can be used to mislead real-world text processing systems. Similarly, attacks could be directed against spam filters, hate speech detection systems, or question answering models. Another example is in medical chatbots where subtly changing a patient's description of symptoms might cause the chatbot to misdiagnose the condition.

## 3) Python method (if possible)

Here's a simplified example using the TextAttack library, demonstrating a word substitution attack. TextAttack provides tools for crafting and defending against adversarial attacks on text. Note that you'll need to install TextAttack and its dependencies.

```python
import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerAttack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load a dataset (e.g., movie review sentiment dataset from HuggingFace)
dataset = HuggingFaceDataset("rotten_tomatoes", None, split="test")

# 2. Load a pre-trained model and tokenizer (e.g., a BERT model fine-tuned for sentiment analysis)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Create a TextAttack model wrapper
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# 4. Choose an attack recipe (e.g., TextFooler)
attack = TextFoolerAttack.build(model_wrapper)

# 5. Construct the TextAttack attacker
attacker = textattack.Attacker(attack, dataset[0])  # Attack on the first sample

# 6. Perform the attack
attack_result = attacker.attack(dataset[0])

# 7. Print the results
print(attack_result)
```

**Explanation:**

1.  **Load Dataset:** Loads a dataset for sentiment analysis.
2.  **Load Model and Tokenizer:** Loads a pre-trained BERT model and its tokenizer from HuggingFace Transformers. The `model_name` specifies the model to use.
3.  **Create Model Wrapper:** Creates a TextAttack model wrapper to interface between the TextAttack library and the Hugging Face model.
4.  **Choose Attack Recipe:** Selects the TextFooler attack recipe, which is a word substitution-based attack.
5.  **Construct Attacker:** Creates a TextAttack Attacker object, specifying the attack recipe and the dataset sample to attack.
6.  **Perform Attack:** Executes the attack on the chosen sample.
7.  **Print Results:** Prints the attack result, which includes the original text, the adversarial text, and the model's predictions for both.

This is a basic example. TextAttack offers various attack strategies (character-level, word-level, etc.), constraint mechanisms (e.g., synonym replacement), and evaluation metrics. You can customize the attack by choosing different attack recipes, models, and datasets.  Also, note that the first run of this code will download the model and tokenizer, and the dataset.

## 4) Follow-up question

How can we defend against adversarial attacks on text models, and what are the trade-offs between different defense strategies (e.g., adversarial training, input sanitization, robust optimization)?
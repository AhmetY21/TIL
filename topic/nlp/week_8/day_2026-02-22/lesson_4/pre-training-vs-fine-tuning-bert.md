---
title: "Pre-training vs Fine-tuning BERT"
date: "2026-02-22"
week: 8
lesson: 4
slug: "pre-training-vs-fine-tuning-bert"
---

# Topic: Pre-training vs Fine-tuning BERT

## 1) Formal definition (what is it, and how can we use it?)

BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model developed by Google. The **pre-training vs. fine-tuning** paradigm is crucial to BERT's effectiveness.

*   **Pre-training:** This is the initial stage where BERT is trained on a massive amount of unlabeled text data (like books and Wikipedia). The model learns general-purpose language representations by completing two unsupervised tasks:

    *   **Masked Language Modeling (MLM):** Randomly masking some of the words in the input and training the model to predict the masked words based on the context.
    *   **Next Sentence Prediction (NSP):** Training the model to predict whether two given sentences are consecutive in the original document. While NSP's utility has been debated and replaced in later models like RoBERTa, it was a core component of original BERT.

    The goal of pre-training is to learn a good general understanding of language, including syntax, semantics, and relationships between words and sentences. The output of pre-training is a model with learned weights and parameters that capture these general language patterns. We *use* the pre-trained model as a starting point.

*   **Fine-tuning:** This is the second stage where the pre-trained BERT model is adapted to a specific downstream task (e.g., sentiment analysis, question answering, text classification). In fine-tuning, we take the pre-trained BERT model and add a task-specific layer (or layers) on top. Then, we train the entire model (or a portion of it) on a labeled dataset relevant to the specific task. The pre-trained weights provide a strong initialization, allowing the model to learn the task-specific nuances with much less data and training time than training from scratch. We *use* the fine-tuned model to perform the task it was specifically trained for.

In essence, pre-training teaches BERT general language knowledge, and fine-tuning applies that knowledge to a specific problem.

## 2) Application scenario

Let's consider the scenario of **sentiment analysis** on customer reviews.

*   **Pre-training:** BERT has already been pre-trained on vast amounts of general text data, giving it a strong grasp of English grammar, vocabulary, and common sense. It understands word relationships and sentence structure.

*   **Fine-tuning:** We have a labeled dataset of customer reviews, where each review is labeled as positive, negative, or neutral.  We take the pre-trained BERT model and add a classification layer on top. We then fine-tune the entire model using our labeled review dataset. During fine-tuning, BERT adjusts its pre-trained weights to better represent the specific linguistic patterns associated with positive, negative, and neutral sentiment in customer reviews. For example, it might learn that certain words or phrases are strongly indicative of positive or negative sentiment in this specific domain.

After fine-tuning, we can use the fine-tuned BERT model to predict the sentiment of new, unseen customer reviews. Because the model started from a good pre-trained state, it will likely perform much better than a model trained from scratch on the same limited review dataset. The model has "transferred" the knowledge learned during pre-training to this new, specific task.

## 3) Python method (if possible)

Here's an example using the Hugging Face Transformers library, which is the standard way to interact with BERT models in Python:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
import torch

# 1. Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) #3 labels: pos, neg, neutral

# 2. Sample data (replace with your actual data)
texts = ["This movie was great!", "I hated this movie.", "It was okay."]
labels = [1, 0, 2]  # 1: positive, 0: negative, 2: neutral

# 3. Prepare data for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

max_length = 128  # Adjust as needed
dataset = SentimentDataset(texts, labels, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. Fine-tuning
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)  # Adjust learning rate as needed

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Fine-tuning complete!")

# 5. Inference (example)
model.eval()  # Set the model to evaluation mode
new_text = "The product is amazing and I love it."
encoding = tokenizer.encode_plus(
            new_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad(): # Disable gradient calculation during inference
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1).item()

print(f"Sentiment prediction: {predictions}") # will give 0,1 or 2 based on our model
```

**Explanation:**

1.  **Load Pre-trained Model and Tokenizer:** We load the pre-trained BERT model (`bert-base-uncased`) and its corresponding tokenizer.  The `BertForSequenceClassification` model is used because we are doing a classification task. `num_labels` specifies how many classes we have (positive, negative, and neutral in this example).

2.  **Sample Data:** This section creates some sample text and label data.  You would replace this with your real dataset.

3.  **Prepare Data for BERT:** The `SentimentDataset` class handles the data preparation. It tokenizes the text using the BERT tokenizer, pads/truncates sequences to a fixed length, and creates PyTorch tensors. The `DataLoader` then loads the data in batches.

4.  **Fine-tuning:** This is the core part.  We move the model to the device (GPU if available, otherwise CPU). We define an optimizer (AdamW is commonly used) and set a learning rate. The code then iterates through the data for a specified number of epochs. In each epoch, it feeds the data to the model, calculates the loss, performs backpropagation to update the model's weights, and updates the optimizer. The `model.train()` and `model.eval()` set the model for training and inference, respectively. The `torch.no_grad()` block is important during inference to prevent the calculation of gradients which is uncessary.

5. **Inference** After training is complete, you can use the fine-tuned model for inference. A new text is tokenized and passed to the model for prediction. The `torch.argmax` function gives the predicted class.

**Important Notes:**

*   This is a simplified example.  For real-world applications, you'll need to handle larger datasets, use more sophisticated data preprocessing techniques, and tune hyperparameters.
*   The choice of `model_name` (e.g., `bert-base-uncased`) depends on your needs.  Larger models generally perform better but require more resources.
*   The learning rate and other hyperparameters should be tuned based on your specific task and dataset.
* You need to have `transformers` and `torch` installed: `pip install transformers torch`

## 4) Follow-up question

How does the performance of a fine-tuned BERT model compare to other approaches like training a simpler model (e.g., logistic regression or a basic neural network) from scratch on the same downstream task data?  Specifically, what are the trade-offs in terms of data requirements, computational resources, and expected accuracy?
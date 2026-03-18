---
title: "The Future of NLP: Multimodal Models"
date: "2026-03-18"
week: 12
lesson: 3
slug: "the-future-of-nlp-multimodal-models"
---

# Topic: The Future of NLP: Multimodal Models

## 1) Formal definition (what is it, and how can we use it?)

Multimodal NLP refers to the area of natural language processing that goes beyond text-only inputs and incorporates other modalities like images, audio, video, and sensor data. Instead of relying solely on textual information, multimodal models aim to understand and generate content that integrates and reasons across multiple modalities.

*   **What it is:** It's an extension of traditional NLP that allows AI models to process and understand data from various sources (e.g., text and images). Think of it as teaching a machine to "see," "hear," and "read" simultaneously to gain a more comprehensive understanding of the world. These models often involve sophisticated architectures that can learn joint representations of different modalities. Common architectures include attention mechanisms, transformers, and graph neural networks, adapted to handle the unique characteristics of each modality.

*   **How can we use it:**

    *   **Improved Understanding:** Multimodal models can understand nuances and context that are difficult or impossible to grasp from text alone. For example, sarcasm might be detected through facial expressions in a video, even if the text itself doesn't explicitly indicate it.
    *   **Content Generation:** Generating richer and more relevant content. Imagine a model that can write a caption for an image, generate a script based on a video, or answer questions about a scene presented visually and verbally.
    *   **Cross-Modal Retrieval:** Searching for information across different modalities. For example, finding a video clip that matches a textual description or identifying an image that corresponds to a specific audio excerpt.
    *   **Contextual Analysis:** Provides the ability to analyze and interpret situations within their broader context. Combining data points from varied modalities ensures a more holistic perception, facilitating greater accuracy in tasks like sentiment analysis.

## 2) Application scenario

**Scenario:** Imagine a customer support chatbot that assists users with smart home appliances.

*   **Traditional (Text-Only) Chatbot:** A user types: "My smart bulb is not turning on." The chatbot might offer standard troubleshooting steps like checking the power supply, the Wi-Fi connection, or resetting the bulb.

*   **Multimodal Chatbot:** The user uploads a picture of the smart bulb showing a blinking red light. The chatbot, using image recognition, identifies the blinking red light pattern. By associating that specific light pattern with a database of error codes and user manuals, the chatbot immediately provides a more precise solution: "The blinking red light indicates a firmware update failure. Please follow these steps to manually update the firmware via the mobile app."

In this scenario, the multimodal chatbot offers a significantly better customer experience because it can *see* the problem, interpret it accurately, and offer a tailored solution based on the visual information, which would be impossible with text alone. Another modality could be audio - the user describes the problem verbally, and the chatbot analyzes the tone and prosody to gauge frustration levels, potentially prioritizing the issue or offering a more empathetic response.

## 3) Python method (if possible)

While building a full multimodal model from scratch requires significant effort and specialized libraries (like PyTorch, TensorFlow, or Hugging Face Transformers), we can illustrate a simplified version using existing pre-trained models. This example shows how you might combine the outputs of a text encoder and an image encoder.

```python
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests
import torch

# Load pre-trained text model (e.g., BERT)
text_model_name = "bert-base-uncased"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

# Load pre-trained image model (e.g., ViT - Vision Transformer)
image_model_name = "google/vit-base-patch16-224"
image_model = AutoModel.from_pretrained(image_model_name)

# Function to process text
def process_text(text):
  inputs = text_tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    outputs = text_model(**inputs)
  return outputs.last_hidden_state.mean(dim=1)  # Average pooling

# Function to process image
def process_image(image_url):
  try:
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((224, 224)) #ViT requires this size
    # Using ViT Feature Extractor (not provided out of box, needs to be installed, but conceptually like tokenizer)
    from transformers import ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(image_model_name)
    inputs = feature_extractor(images=image, return_tensors="pt")


    with torch.no_grad():
      outputs = image_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) # Average pooling
  except Exception as e:
    print(f"Error processing image: {e}")
    return None

# Example usage
text = "A cute cat sitting on a mat."
image_url = "https://example.com/cat.jpg" # Replace with an actual image URL
# This URL will likely not work, but demonstrates the URL loading for images


text_embedding = process_text(text)
image_embedding = process_image(image_url)

# Combine embeddings (e.g., concatenation or element-wise addition)
if text_embedding is not None and image_embedding is not None:
  combined_embedding = torch.cat((text_embedding, image_embedding), dim=1) # Simple concatenation

  print("Shape of combined embedding:", combined_embedding.shape) #Should be [1, 768 + 768] if both models produce 768 sized embeddings
else:
  print("Failed to generate embedding(s).")
```

**Explanation:**

1.  **Load Pre-trained Models:**  The code uses the Hugging Face `transformers` library to load pre-trained BERT (text) and ViT (image) models. These models have already been trained on large datasets and can extract meaningful features from text and images.
2.  **Process Text:** The `process_text` function tokenizes the input text using the BERT tokenizer and feeds it to the BERT model. The function then takes the mean of the hidden states as the text embedding.
3.  **Process Image:** The `process_image` function loads an image from a URL using the `PIL` library.  The image is resized to 224x224 for ViT compatibility. ViTFeatureExtractor prepares the image for the ViT model.  The function returns the mean of the hidden states as the image embedding.
4.  **Combine Embeddings:** The code concatenates the text and image embeddings to create a combined multimodal representation.  More sophisticated methods like attention mechanisms could be used to weight the modalities differently.

**Important Notes:**

*   You need to install the `transformers` and `PIL` libraries: `pip install transformers pillow requests`
*   Replace `"https://example.com/cat.jpg"` with a working URL of an image.
*   This is a simplified example.  Real-world multimodal models are significantly more complex, involving careful training and fine-tuning on multimodal datasets.
*   Consider using GPU acceleration if you're working with large images or models.
* The ViTFeatureExtractor needs to be installed using `pip install timm`. This wasn't directly stated in many ViT tutorials.

## 4) Follow-up question

How can we evaluate the performance of multimodal models, given that the evaluation metrics for individual modalities (e.g., BLEU for text, accuracy for image classification) might not directly translate to the combined performance?  What novel evaluation metrics are being developed specifically for multimodal tasks?
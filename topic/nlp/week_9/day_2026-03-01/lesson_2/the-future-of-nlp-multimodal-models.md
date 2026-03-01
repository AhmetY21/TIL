---
title: "The Future of NLP: Multimodal Models"
date: "2026-03-01"
week: 9
lesson: 2
slug: "the-future-of-nlp-multimodal-models"
---

# Topic: The Future of NLP: Multimodal Models

## 1) Formal definition (what is it, and how can we use it?)

Multimodal models in NLP refer to models that process and integrate information from multiple modalities (i.e., different forms of data), such as text, images, audio, video, and sensor data. Traditional NLP models primarily focus on text, while multimodal models aim to understand the world by leveraging the complementary information contained in these diverse modalities.

*   **What is it?** A multimodal model learns joint representations that capture the relationships between different modalities. It's not simply about processing each modality independently; it's about understanding how they relate to each other. For example, a model could learn that a picture of a cat is related to the word "cat" and the sound "meow."

*   **How can we use it?** We can use multimodal models for various tasks that require understanding information from multiple sources, including:

    *   **Visual Question Answering (VQA):** Answering questions about an image (e.g., "What color is the cat?" given an image of a cat).
    *   **Image Captioning:** Generating textual descriptions of images.
    *   **Video Understanding:** Analyzing videos to understand events, activities, and relationships.
    *   **Sentiment Analysis:** Determining the sentiment expressed in a video, considering both visual cues (facial expressions, body language) and audio cues (tone of voice, spoken words).
    *   **Cross-modal Retrieval:** Searching for images based on textual descriptions, or vice versa.
    *   **Robotics:** Enabling robots to interact with the world by understanding both visual and auditory inputs.
    *   **Healthcare:** Analyzing medical images and patient records to diagnose diseases.

The key advantage of multimodal models is their ability to leverage richer, more complete information, leading to more accurate and robust performance compared to unimodal models. They can handle ambiguity present in a single modality by using information from other modalities as context. For instance, an ambiguous sentence can be understood more effectively with the context of a corresponding image.

## 2) Application scenario

Let's consider the application scenario of **Video Understanding for Action Recognition.**

Imagine a self-driving car trying to understand a street scene. It needs to not only identify objects (cars, pedestrians, traffic lights) but also understand their actions and intentions. A unimodal NLP approach focusing solely on text descriptions from traffic reports would be insufficient.

A multimodal model could analyze video footage from the car's cameras, incorporating both visual and audio information.

*   **Visual Input:** The model would analyze the video frames to identify objects and track their movements.  It might detect a pedestrian walking towards the crosswalk.
*   **Audio Input:**  The model would analyze audio signals for sounds like sirens, horns, or spoken warnings.  It might hear a siren approaching.
*   **Textual Input:** Using sensors in the area, the model might receive textual descriptions of events such as warnings, or construction alerts.

By integrating information from these modalities, the model could better understand the overall situation. For example, detecting a pedestrian moving towards a crosswalk while simultaneously hearing a siren increases the likelihood that the car needs to slow down and be prepared to stop.  This combined understanding allows the car to react more safely and intelligently than if it were relying solely on vision or sound alone. The model could even generate a textual summary of the scene, such as "Pedestrian approaching crosswalk, possible emergency vehicle approaching."

## 3) Python method (if possible)

While implementing a full multimodal model from scratch is complex, we can demonstrate a simplified example using transformers and pre-trained models to illustrate the basic concepts.  We'll use `transformers` library with `CLIP` model for image and text feature extraction and a simple concatenation for multimodal representation.

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-base")
processor = CLIPProcessor.from_pretrained("openai/clip-base")

# Define image and text inputs
url = "http://images.cocodataset.org/val2017/000000039769.jpg" #example image
image = Image.open(requests.get(url, stream=True).raw)
text = "a cat sitting on a mat"

# Process inputs using the CLIP processor
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

# Extract image and text embeddings
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds

# Concatenate embeddings to create a multimodal representation
multimodal_embedding = torch.cat((image_embeddings, text_embeddings), dim=1)

# Print the shape of the multimodal embedding
print("Shape of multimodal embedding:", multimodal_embedding.shape)

# You can now use this multimodal_embedding for downstream tasks, such as classification or retrieval
# Note: This is a simplified example. More sophisticated methods exist for fusing multimodal information,
# such as attention mechanisms and learnable fusion layers.
```

**Explanation:**

1.  **Load Pre-trained Models:** We use the `transformers` library to load a pre-trained CLIP (Contrastive Language-Image Pre-training) model. CLIP is trained to align image and text representations, making it suitable for multimodal tasks.
2.  **Prepare Inputs:** We load an image and a corresponding text description.  The `CLIPProcessor` is used to preprocess both inputs into a format that the model can understand (tokenizing the text and resizing the image).
3.  **Extract Embeddings:** We pass the processed inputs to the CLIP model to obtain image and text embeddings. These embeddings are vector representations of the image and text.
4.  **Fuse Embeddings:** We concatenate the image and text embeddings along the dimension to create a multimodal embedding. This combined embedding represents both the visual and textual information.

This example demonstrates a simple method for creating a multimodal representation. However, there are more advanced techniques, such as attention mechanisms and learnable fusion layers, that can improve performance.

## 4) Follow-up question

How can we evaluate the performance of a multimodal model in a task like Video Question Answering, considering that the evaluation needs to account for the interaction between different modalities?  What are some common metrics used, and what challenges do they address?
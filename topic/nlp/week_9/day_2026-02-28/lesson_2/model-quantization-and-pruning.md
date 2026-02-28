---
title: "Model Quantization and Pruning"
date: "2026-02-28"
week: 9
lesson: 2
slug: "model-quantization-and-pruning"
---

# Topic: Model Quantization and Pruning

## 1) Formal definition (what is it, and how can we use it?)

**Model Quantization:**

Quantization is a technique that reduces the precision of the weights and/or activations in a neural network. Typically, deep learning models use 32-bit floating-point numbers (FP32) to represent these values. Quantization converts these to lower-precision formats, such as 16-bit floating-point (FP16), 8-bit integers (INT8), or even binary (1-bit).

* **What is it?** A process of converting the numerical representation of a neural network's parameters (weights and activations) from a higher-precision format (e.g., FP32) to a lower-precision format (e.g., INT8).  This is a form of model compression.
* **How can we use it?** Quantization reduces model size, improves inference speed, and lowers power consumption, making models more suitable for deployment on resource-constrained devices like mobile phones, embedded systems, and edge devices. It also allows for faster computations since lower-precision arithmetic operations are typically more efficient.

**Model Pruning:**

Pruning is a technique that removes less important connections (weights) in a neural network. This results in a sparse model, meaning many weights are set to zero.

* **What is it?** A process of removing connections (weights) from a neural network, typically by setting them to zero based on some importance criteria.
* **How can we use it?** Pruning reduces model size, improves inference speed, and potentially improves generalization performance by reducing overfitting. The resulting sparse model requires less memory and fewer computations, which is beneficial for deployment on devices with limited resources. Pruning also increases energy efficiency.

In combination, quantization and pruning can dramatically reduce the size and computational cost of deep learning models without significantly sacrificing accuracy.  They are key techniques in model optimization for deployment.

## 2) Application scenario

Let's consider an application scenario involving a large language model (LLM) used for sentiment analysis on a mobile phone.

*   **Problem:**  A pre-trained, highly accurate LLM in FP32 format is too large to fit on the phone and too computationally expensive to run in real-time. This leads to slow response times and excessive battery drain.
*   **Solution:**
    *   **Quantization:** Apply quantization to convert the model's weights and activations from FP32 to INT8. This reduces the model size by a factor of 4 (approximately) and allows the mobile phone's CPU/GPU to perform faster integer arithmetic.
    *   **Pruning:** Prune the model to remove less important connections. This further reduces the model size and computational complexity, improving inference speed. A sparsity of, for example, 50% to 90% might be achievable without significant loss in accuracy.
*   **Benefits:** The combined effect of quantization and pruning results in a smaller, faster, and more energy-efficient LLM that can be deployed on the mobile phone, enabling real-time sentiment analysis without negatively impacting the user experience or battery life. Other examples of application scenarios include:

    *   **Edge computing:** Deploying computer vision models for object detection on smart cameras.
    *   **Embedded systems:**  Running speech recognition models on microcontrollers.
    *   **Server-side inference:** Reducing the cost and latency of serving models in the cloud.

## 3) Python method (if possible)

TensorFlow and PyTorch provide tools for quantization and pruning. Here's an example using TensorFlow:

```python
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity import keras as sparsity

# Assume you have a Keras model called 'model'

# Example model (replace with your actual model)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
  tf.keras.layers.Dense(1)
])

# Pruning
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(
          initial_sparsity=0.50,
          final_sparsity=0.90,
          begin_step=2000,
          end_step=10000,
          frequency=100
      )
  }

model_for_pruning = prune.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Train the pruned model
logdir = '/tmp/log'
callbacks = [
  pruning_callbacks.UpdatePruningStep(),
  pruning_callbacks.PruningSummaries(log_dir=logdir, profile_model=True)
]

# Sample data (replace with your actual data)
import numpy as np
x_train = np.random.rand(1000, 5)
y_train = np.random.rand(1000, 1)

model_for_pruning.fit(x_train, y_train,
                  batch_size=32, epochs=2,
                  callbacks=callbacks)


# Strip pruning wrappers for inference
model_for_export = sparsity.strip_pruning(model_for_pruning)

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
# Provide a representative dataset (this is crucial for INT8 quantization!)
def representative_data_gen():
  for i in range(100):
    yield [np.random.rand(1, 5).astype(np.float32)]  # Replace with your actual input shape and data type

converter.representative_dataset = representative_data_gen

tflite_quantized_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
  f.write(tflite_quantized_model)

```

**Explanation:**

*   **Pruning:**  The code uses `tensorflow_model_optimization` to prune the model. It defines a pruning schedule that gradually increases the sparsity of the model during training. The `prune_low_magnitude` function applies pruning based on weight magnitudes.  The `UpdatePruningStep` and `PruningSummaries` callbacks manage the pruning process during training. Finally `sparsity.strip_pruning` removes pruning related operations from the model, creating a sparse model suitable for deployment.
*   **Quantization:** The code uses `tf.lite.TFLiteConverter` to convert the pruned model to a TensorFlow Lite model.  The `optimizations` parameter is set to `tf.lite.Optimize.DEFAULT` to enable post-training quantization.  For INT8 quantization, a `representative_dataset` must be provided. This dataset is used to calibrate the quantization process and determine the optimal scaling factors.

**Note:**  This is a simplified example. The specific pruning schedule and quantization parameters may need to be tuned to achieve the best performance for your specific model and application. The `representative_data_gen` function needs to be replaced with your actual training data samples.
PyTorch has similar functionality, including pruning via the `torch.nn.utils.prune` module and quantization tools available in `torch.quantization`.

## 4) Follow-up question

How does the choice of pruning strategy (e.g., weight magnitude pruning, gradient-based pruning, random pruning) affect the accuracy and speedup achieved after pruning and quantization? How do you determine the optimal balance between model size reduction and accuracy preservation during the pruning and quantization process?
# Teachable Machine Date Classifier (Keras)

This repository contains a trained image recognition model built using [Teachable Machine by Google](https://teachablemachine.withgoogle.com/). The model is exported in **TensorFlow Keras (.h5)** format.

## 🧠 Model Description

The classifier is trained to recognize **four types of dates**:
- Sukkary
- Ajwa
- Barhi
- Medjool

## 📁 Files Included

- `keras_model.h5` — The trained image recognition model in Keras format.
- `labels.txt` — Label list (class index to name mapping).

## 📦 How to Use

To make predictions using this model, load it with TensorFlow:

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model("keras_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocess image
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Predict
image = preprocess_image("your_test_image.jpg")
prediction = model.predict(image)
predicted_label = labels[np.argmax(prediction)]

print("Predicted class:", predicted_label)

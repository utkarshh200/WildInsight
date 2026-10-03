# 🐾 WildInsight

### Animal Species Recognition using Deep Learning

WildInsight is a deep-learning-based animal species recognition system that identifies animals from uploaded images.

The current version recognizes four animal classes:

- 🐆 Leopard
- 🦁 Lion
- 🐅 Tiger
- 🐺 Wolf

The project uses a ResNet-18 convolutional neural network and provides prediction confidence, class probabilities, alternative predictions for uncertain images, and Grad-CAM visual explanations.

---

## 🎯 Objective

The objective of WildInsight is to build an image-based animal recognition system that can:

1. Identify an animal species from an image.
2. Provide the model's confidence for its prediction.
3. Show probabilities for all supported classes.
4. Provide an alternative prediction when the model is uncertain.
5. Visualize the image regions influencing the prediction using Grad-CAM.

---

## 🧠 Model

WildInsight uses **ResNet-18**, a convolutional neural network architecture designed for image recognition.

A pretrained ResNet architecture was adapted and trained/fine-tuned for four wildlife classes:

```text
Input Image
     ↓
Image Preprocessing
     ↓
ResNet-18
     ↓
4-Class Classification
     ↓
Leopard / Lion / Tiger / Wolf
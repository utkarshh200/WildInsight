import torch
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image


# -----------------------------
# 1. Animal classes
# -----------------------------

classes = ["leopard", "lion", "tiger", "wolf"]


# -----------------------------
# 2. Load our trained model
# -----------------------------

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        "wildinsight_model.pth",
        map_location="cpu"
    )
)

model.eval()


# -----------------------------
# 3. Load the image
# -----------------------------

image_path = "image.tiger.jpg"

image = Image.open(image_path).convert("RGB")


# -----------------------------
# 4. Prepare the image
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_tensor = transform(image).unsqueeze(0)


# -----------------------------
# 5. Capture the final
#    convolutional feature maps
# -----------------------------

activations = []
gradients = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer = model.layer4

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# -----------------------------
# 6. Make prediction
# -----------------------------

output = model(image_tensor)

prediction = output.argmax(dim=1).item()

print("Prediction:", classes[prediction])


# -----------------------------
# 7. Calculate Grad-CAM
# -----------------------------

model.zero_grad()

output[0, prediction].backward()

activation = activations[0]
gradient = gradients[0]


# Average gradients across the feature-map dimensions
weights = gradient.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (weights * activation).sum(dim=1)

cam = torch.relu(cam)

cam = cam.squeeze()

# Normalize between 0 and 1
cam = cam - cam.min()

cam = cam / (cam.max() + 1e-8)

cam = cam.detach().numpy()


# -----------------------------
# 8. Display original image
#    and Grad-CAM
# -----------------------------

import numpy as np
import cv2

# Convert Grad-CAM to 0-255
cam_image = np.uint8(255 * cam)

# Resize heatmap to original image size
cam_image = cv2.resize(
    cam_image,
    (image.width, image.height)
)

# Apply a color map
heatmap = cv2.applyColorMap(
    cam_image,
    cv2.COLORMAP_JET
)

# Convert RGB image to NumPy
original = np.array(image)

# Convert OpenCV BGR → RGB
heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)

# Combine original image and heatmap
overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)

# Display
plt.figure(figsize=(10, 7))

plt.imshow(overlay)

plt.axis("off")

plt.title(
    f"WildInsight Grad-CAM: {classes[prediction].upper()}"
)

plt.show()
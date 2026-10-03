import torch
from torchvision import models, transforms
from PIL import Image

# 1. Animal classes
classes = ["leopard", "lion", "tiger", "wolf"]

# 2. Load the trained model
model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load("wildinsight_model.pth", map_location="cpu")
)

model.eval()

# 3. Prepare the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

import sys

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")

image = transform(image).unsqueeze(0)

# 4. Ask the model for a prediction
with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

# 5. Show the result
animal = classes[prediction.item()]

print("Prediction:", animal)
print("Confidence:", confidence.item() * 100, "%")
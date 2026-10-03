from ultralytics import YOLOWorld
from PIL import Image
import torch
from torchvision import models, transforms

# -----------------------------
# 1. Load YOLO-World
# -----------------------------
detector = YOLOWorld("yolov8s-world.pt")

detector.set_classes([
    "animal",
    "leopard",
    "lion",
    "tiger",
    "wolf"
])

image_path = "test_images/Leopard_hiding.jpg"

results = detector.predict(
    image_path,
    conf=0.01,
    imgsz=1280
)

result = results[0]

# -----------------------------
# 2. Find the largest detection
# -----------------------------
if len(result.boxes) == 0:
    print("No animal detected.")
    exit()

# -----------------------------
# 2. Find the best animal detection
# -----------------------------

best_box = None
best_confidence = 0

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    class_name = detector.names[class_id]

    print(
        f"Detected: {class_name} "
        f"| Confidence: {confidence:.3f}"
    )

    # Prefer leopard detection
    if class_name == "leopard" and confidence > best_confidence:
        best_confidence = confidence
        best_box = box.xyxy[0].tolist()

# If no leopard was found, use the highest-confidence detection
if best_box is None:

    print("No leopard detected. Using highest-confidence detection.")

    best_box = result.boxes[0].xyxy[0].tolist()

x1, y1, x2, y2 = best_box

# -----------------------------
# 3. Crop the animal
# -----------------------------
image = Image.open(image_path).convert("RGB")

x1, y1, x2, y2 = best_box

crop = image.crop((
    int(x1),
    int(y1),
    int(x2),
    int(y2)
))

crop.save("animal_crop.jpg")

print("Animal crop saved as: animal_crop.jpg")

# -----------------------------
# 4. Load our ResNet-18
# -----------------------------
classes = [
    "leopard",
    "lion",
    "tiger",
    "wolf"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        "wildinsight_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# 5. Prepare crop
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(crop)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

# -----------------------------
# 6. Predict
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)

probabilities = torch.softmax(output, dim=1)[0]

predicted_index = torch.argmax(probabilities).item()

print()
print("========== RESNET RESULT ==========")
print(
    f"Prediction: {classes[predicted_index]}"
)

print(
    f"Confidence: "
    f"{probabilities[predicted_index].item() * 100:.2f}%"
)

print()
print("Probabilities:")

for i, class_name in enumerate(classes):
    print(
        f"{class_name}: "
        f"{probabilities[i].item() * 100:.2f}%"
    )
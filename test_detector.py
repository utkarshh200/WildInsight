from ultralytics import YOLOWorld
from PIL import Image

# Load YOLO-World
model = YOLOWorld("yolov8s-world.pt")

# Tell the detector what we're looking for
model.set_classes([
    "animal",
    "mammal",
    "leopard",
    "lion",
    "tiger",
    "wolf"
])

# Test image
image_path = "test_images/image_lion.webp"

# Detect
results = model.predict(
    image_path,
    conf=0.05
)

# Show results
results[0].show()

# Print detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        f"Detected: {model.names[class_id]} "
        f"| Confidence: {confidence:.2f}"
    )
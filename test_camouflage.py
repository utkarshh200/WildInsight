from PIL import Image
from camouflage_detector import detect_and_crop


image_path = "test_images/image_lion.webp"

image = Image.open(image_path).convert("RGB")

crop, detected, label, confidence = detect_and_crop(image)

if detected:

    print("Animal detected!")
    print(f"YOLO label: {label}")
    print(f"YOLO confidence: {confidence * 100:.2f}%")

    crop.save("camouflage_crop.jpg")

    print("Crop saved as: camouflage_crop.jpg")

else:

    print("No animal detected.")
    print("Using original image as fallback.")
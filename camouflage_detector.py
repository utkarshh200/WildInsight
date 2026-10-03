from ultralytics import YOLOWorld
from PIL import Image

import torch
from torchvision import models, transforms


# -----------------------------------
# Load YOLO-World
# -----------------------------------

detector = YOLOWorld("yolov8s-world.pt")

detector.set_classes([
    "animal",
    "leopard",
    "lion",
    "tiger",
    "wolf"
])


# -----------------------------------
# Load WildInsight ResNet
# -----------------------------------

classes = [
    "leopard",
    "lion",
    "tiger",
    "wolf"
]

classifier = models.resnet18(weights=None)

classifier.fc = torch.nn.Linear(
    classifier.fc.in_features,
    4
)

classifier.load_state_dict(
    torch.load(
        "wildinsight_model.pth",
        map_location="cpu"
    )
)

classifier.eval()


classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------------
# Detect + classify candidate crops
# -----------------------------------

def detect_and_crop(image):

    results = detector.predict(
        image,
        conf=0.05,
        imgsz=1280,
        verbose=False
    )

    result = results[0]

    if len(result.boxes) == 0:
        return image, False, None, 0.0


    candidates = []

    image_width, image_height = image.size
    image_area = image_width * image_height


    # -----------------------------------
    # Create candidate crops
    # -----------------------------------

    for box in result.boxes:

        detector_confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        detector_label = detector.names[
            class_id
        ]

        # Ignore extremely weak detections
        if detector_confidence < 0.05:
            continue


        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image_width, int(x2))
        y2 = min(image_height, int(y2))


        if x2 <= x1 or y2 <= y1:
            continue


        # Small padding around detected animal
        padding_x = (x2 - x1) * 0.05
        padding_y = (y2 - y1) * 0.05

        x1 = max(
            0,
            int(x1 - padding_x)
        )

        y1 = max(
            0,
            int(y1 - padding_y)
        )

        x2 = min(
            image_width,
            int(x2 + padding_x)
        )

        y2 = min(
            image_height,
            int(y2 + padding_y)
        )


        crop = image.crop(
            (x1, y1, x2, y2)
        )


        # -----------------------------------
        # ResNet evaluates this crop
        # -----------------------------------

        tensor = classifier_transform(
            crop
        ).unsqueeze(0)


        with torch.no_grad():

            output = classifier(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            prediction = torch.argmax(
                probabilities
            ).item()

            resnet_confidence = float(
                probabilities[prediction]
            )


        predicted_species = classes[
            prediction
        ]


        # -----------------------------------
        # Calculate crop size
        # -----------------------------------

        crop_area = (
            (x2 - x1)
            *
            (y2 - y1)
        )

        area_ratio = (
            crop_area / image_area
        )


        # -----------------------------------
        # Candidate score
        # -----------------------------------
        #
        # ResNet confidence is most important.
        # Crop size gives a small preference to
        # a larger/main animal when confidence
        # is similar.
        #

        area_factor = min(
            area_ratio * 4,
            1.0
        )

        score = (
            resnet_confidence * 0.85
            +
            area_factor * 0.15
        )


        candidates.append({
            "crop": crop,
            "detector_label": detector_label,
            "detector_confidence": detector_confidence,
            "prediction": predicted_species,
            "resnet_confidence": resnet_confidence,
            "score": score
        })


    # -----------------------------------
    # No useful candidate
    # -----------------------------------

    if not candidates:

        return image, False, None, 0.0


    # -----------------------------------
    # Select best candidate
    # -----------------------------------

    best = max(
        candidates,
        key=lambda x: x["score"]
    )


    return (
        best["crop"],
        True,
        best["prediction"],
        best["detector_confidence"]
    )
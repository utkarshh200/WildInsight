import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# =========================
# 1. Prepare test images
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = ImageFolder(
    "wildlife_data/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# =========================
# 2. Load trained model
# =========================

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

classes = test_dataset.classes

print("Classes:", classes)


# =========================
# 3. Test the model
# =========================

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        output = model(images)

        probabilities = torch.softmax(output, dim=1)

        prediction = torch.argmax(
            probabilities,
            dim=1
        )

        actual = labels.item()
        predicted = prediction.item()

        all_labels.append(actual)
        all_predictions.append(predicted)

        confidence = (
            probabilities[0][predicted].item()
            * 100
        )

        image_path = test_dataset.samples[total][0]
        filename = image_path.split("\\")[-1]

        actual_name = classes[actual]
        predicted_name = classes[predicted]

        if actual == predicted:
            symbol = "✅"
            correct += 1
        else:
            symbol = "❌"

        print(
            f"{filename} → "
            f"Actual: {actual_name} | "
            f"Predicted: {predicted_name} "
            f"({confidence:.2f}%) {symbol}"
        )

        total += 1


# =========================
# 4. Overall accuracy
# =========================

accuracy = 100 * correct / total

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Correct predictions:", correct)
print("Total test images:", total)
print(f"Accuracy: {accuracy:.2f}%")


# =========================
# 5. Classification report
# =========================

print("\n==============================")
print("PRECISION / RECALL / F1-SCORE")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=2
    )
)


# =========================
# 6. Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# =========================
# 7. Display Confusion Matrix
# =========================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot()

plt.title("WildInsight - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as: confusion_matrix.png")
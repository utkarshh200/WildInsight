import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# -----------------------------
# 1. Choose the device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# 2. Prepare the images
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# 3. Load our datasets
# -----------------------------

train_dataset = ImageFolder(
    "wildlife_data/train",
    transform=transform
)

validation_dataset = ImageFolder(
    "wildlife_data/validation",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False
)


# -----------------------------
# 4. Show our animal classes
# -----------------------------

print("Animal classes:", train_dataset.classes)


# -----------------------------
# 5. Load pre-trained ResNet-18
# -----------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# -----------------------------
# 6. Replace the final layer
# -----------------------------

model.fc = torch.nn.Linear(
    model.fc.in_features,
    4
)

model = model.to(device)


# -----------------------------
# 7. Training settings
# -----------------------------

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# -----------------------------
# 8. Train the model
# -----------------------------

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {total_loss / len(train_loader):.4f}"
    )


# -----------------------------
# 9. Save our trained model
# -----------------------------

torch.save(
    model.state_dict(),
    "wildinsight_model.pth"
)

print("Training complete!")
print("Model saved as wildinsight_model.pth")
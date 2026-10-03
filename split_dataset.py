import os
import shutil
import random

source = "dataset/animals/animals"
destination = "wildlife_data"

animals = ["tiger", "lion", "leopard", "wolf"]

# Create train, validation and test folders
for split in ["train", "validation", "test"]:
    for animal in animals:
        os.makedirs(
            os.path.join(destination, split, animal),
            exist_ok=True
        )

# Make the results reproducible
random.seed(42)

for animal in animals:

    source_folder = os.path.join(source, animal)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    # Shuffle the images
    random.shuffle(images)

    # Split: 70% train, 15% validation, 15% test
    train_images = images[:42]
    validation_images = images[42:51]
    test_images = images[51:60]

    # Copy the images
    for image in train_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(destination, "train", animal, image)
        )

    for image in validation_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(destination, "validation", animal, image)
        )

    for image in test_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(destination, "test", animal, image)
        )

    print(
        animal,
        "→ Train:", len(train_images),
        "| Validation:", len(validation_images),
        "| Test:", len(test_images)
    )

print("\nDataset split completed!")
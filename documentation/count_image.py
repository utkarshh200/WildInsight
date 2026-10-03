import os

dataset_path = "dataset/animals/animals"

animals = ["tiger", "lion", "leopard", "wolf"]

for animal in animals:
    folder = os.path.join(dataset_path, animal)

    images = [
        file for file in os.listdir(folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(animal, ":", len(images), "images")
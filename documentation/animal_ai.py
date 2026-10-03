import torch
from torchvision import models, transforms
from PIL import Image

#model = models.resnet18(weights="DEFAULT") #Give me ResNet-18, already trained on a huge image dataset.

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.eval()#Telling the model to make predictions now, not to train

labels = weights.meta["categories"]


#image = Image.open("Royal-Bengal-Tiger-1024x767.jpg")
image = Image.open("Gray-wolf.webp")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_tensor = preprocess(image)

#print("Original image size:", image.size)
#print("Prepared image shape:", image_tensor.shape)

image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)

    predicition = torch.argmax(output, dim=1)

    #print("Predicted class index:", predicition.item())
    
    class_number = predicition.item()
    print("Predicted class number:", class_number)
    print("Predicted animal:", labels[class_number])
    
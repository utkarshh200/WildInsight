from PIL import Image

image = Image.open("Royal-Bengal-Tiger-1024x767.jpg")

#print(image.size)

#pixel = image.getpixel((0,0))
#print(pixel)

#image.show()

print("Original size:", image.size)

resized_image = image.resize((224,224)) #Take the original image and make a new version that is 224 × 224 pixels.

print("New size:", resized_image.size)

resized_image.show()
import torch
from torchvision import transforms, models
from PIL import Image


model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(r"C:\projects\PlantAI\model\plant_model.pth"))
model.eval()

classes = ['diseased', 'healthy']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


img_path = r"C:\projects\PlantAI\test.jpg"   
image = Image.open(img_path).convert('RGB')

image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Prediction:", classes[predicted.item()])
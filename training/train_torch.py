import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware routing complete. Using device: {device}")


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.3
    ),
    
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(
    r"C:\projects\PlantAI\Dataset\train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    r"C:\projects\PlantAI\Dataset\val",
    transform=val_transform
)

print(f"Class mapping established: {train_data.class_to_idx}")

# Batch size of 32 is a good standard. If you get an OutOfMemory error, drop it to 16.
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_bar.set_postfix(loss=loss.item())

    epoch_train_loss = running_loss / len(train_data)

  
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model weights
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), r"C:\projects\PlantAI\model\plant_model.pth")
        print("✅ Best model saved!")

print("🔥 Training fully completed!")
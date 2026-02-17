import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
import copy
from tqdm import tqdm

# Config
MODEL_NAME = "sdv5_expert"
DATA_DIR = Path("datasets/genimage/imagenet_ai_0424_sdv5")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir):
    """
    Standard ImageNet normalization + Augmentation for training.
    """
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load dataset
    full_dataset = datasets.ImageFolder(data_dir / "train", transform=data_transforms['train'])
    
    # if standard val folder doesn't exist or we want a custom split
    val_dir = data_dir / "val"
    if val_dir.exists():
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    else:
        # If no val folder, split the training set (80/20)
        print("No 'val' folder found. Splitting training set...")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        # Apply strict validation transforms (no flips) to the val set
        val_dataset.dataset.transform = data_transforms['val']

    # 4. Data Loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes # Should be ['ai', 'nature']
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"Training on {DEVICE}...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save checkpoint immediately
                torch.save(model.state_dict(), f"models/{MODEL_NAME}_best.pth")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # Set directories
    Path("models").mkdir(exist_ok=True)

    # Get data
    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR)
    print(f"Classes: {class_names}") # Verify ['ai', 'nature']
    
    # Setup model
    print("Loading ResNet-50...")
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model_ft.parameters():
        param.requires_grad = False
    
    # Replace the final Fully Connected layer
    # ResNet50 input features for fc is 2048
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2) # Binary: AI vs Nature

    model_ft = model_ft.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Optimize all parameters
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    # Train
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

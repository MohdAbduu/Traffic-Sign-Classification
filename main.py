import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import TrafficSignCNN, TinyVGG3
from visualization import (
     plot_confusion_matrix,
     evaluate_metrics,
     plot_training_history,
     compare_models,
     plot_classification_report,
     show_example_predictions
)
import warnings
warnings.filterwarnings("ignore")

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

# Custom Dataset
class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Validate required columns
        required_columns = ['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path']
        missing_columns = [col for col in required_columns if col not in self.data_frame.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        self.classes = sorted(self.data_frame['ClassId'].unique())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get row data
        row = self.data_frame.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, str(row['Path']))
        image = Image.open(img_path).convert('RGB')

        # Get ROI coordinates
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])

        # Crop image to ROI if coordinates are valid
        if all(v >= 0 for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1:
            try:
                image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Warning: Could not crop image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = int(row['ClassId'])

        return image, label

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load datasets
try:
    train_dataset = TrafficSignDataset(
        csv_file='Data/Train.csv',
        img_dir='Data',
        transform=transform
    )
    print(f"Loaded training dataset with {len(train_dataset)} samples")

    test_dataset = TrafficSignDataset(
        csv_file='Data/Test.csv',
        img_dir='Data',
        transform=transform
    )
    print(f"Loaded test dataset with {len(test_dataset)} samples")

except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)

# Create numerical label map
label_map = {i: str(i) for i in train_dataset.classes}
print(f"Using {len(label_map)} classes")

def train_model(model, model_name, num_epochs=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Evaluation phase
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(correct / total)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]*100:.2f}%, "
              f"Test Acc: {test_accuracies[-1]*100:.2f}%")

    # Final evaluation metrics
    acc, prec, rec, f1 = evaluate_metrics(all_labels, all_preds)
    print(f"\n{model_name} Final Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    plot_confusion_matrix(all_labels, all_preds, label_map, model_name)
    plot_classification_report(all_labels, all_preds, label_map)
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, model_name)

    # Show example predictions
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            show_example_predictions(model, images, labels, label_map, model_name)
            break

    # Save model
    torch.save(model.state_dict(), f"{model_name.lower().replace(' ', '_')}.pth")
    print(f"Model saved as {model_name.lower().replace(' ', '_')}.pth")

    return {
        'name': model_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_accuracies,
        'test_acc': test_accuracies
    }

# Run training for both models
num_classes = len(train_dataset.classes)
print(f"Training models for {num_classes} classes")

# Update the function calls
history_cnn = train_model(TrafficSignCNN(num_classes), "TrafficSignCNN", num_epochs=3)
history_vgg = train_model(TinyVGG3(num_classes), "TinyVGG3", num_epochs=3)
# Compare model performances
compare_models([history_cnn, history_vgg])
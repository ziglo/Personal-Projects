import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import sys

# Import your custom modules
from models.mood_classifier import MoodClassifier
from data.dataset import DogMoodDataset

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define Transforms
# IMPORTANT: Must match the validation transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
dataset = DogMoodDataset(
    image_dir=r'c:\Users\Austin\Desktop\Projects\dog-mood-detector\dogproject\data',
    transform=None
)

# 4. Load Model Checkpoint
# --- FIX: Pointing to the correctly saved v3 model ---
model_path = "dog_mood_regNet_v3.pth"

if not os.path.exists(model_path):
    print(f"ERROR: Could not find {model_path}.")
    print("Did you wait for the training to finish and save?")
    sys.exit()

print(f"Loading NEW model: {model_path}")
checkpoint = torch.load(model_path, map_location=device)
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_classes = checkpoint['num_classes']

print(f"Classes: {label_to_idx}")

# 5. Encode Labels
def encode_labels(labels):
    encoded = []
    for label in labels:
        if isinstance(label, str):
            encoded.append(label_to_idx[label])
        else:
            encoded.append(int(label))
    return encoded

val_labels_encoded = encode_labels(dataset.val_labels)

# 6. Dataset Wrapper
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform: img = self.transform(img)
        return img, label

val_dataset = CustomDataset(dataset.val_images, val_labels_encoded, transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 7. Load Model
model = MoodClassifier(input_size=(3, 224, 224), num_classes=num_classes).model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 8. Evaluation
print("\n=== Evaluating v3 Model ===")

# Matrix: rows = Actual, cols = Predicted
cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Fill Confusion Matrix
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            cm[t.long(), p.long()] += 1

accuracy = 100 * correct / total
print(f'Overall Validation Accuracy: {accuracy:.2f}%')

# 9. Print Text-Based Confusion Matrix
print("\n=== Confusion Matrix ===")
print("(Rows = Actual Label, Columns = Predicted Label)\n")

# Header Row
header = f"{'ACTUAL':<12} |"
for i in range(num_classes):
    header += f" {idx_to_label[i]:<10}"
print(header)
print("-" * len(header))

# Data Rows
for i in range(num_classes):
    actual_label = idx_to_label[i]
    row_str = f"{actual_label:<12} |"
    for j in range(num_classes):
        count = cm[i, j].item()
        row_str += f" {count:<10}"
    print(row_str)

print("\nDone.")
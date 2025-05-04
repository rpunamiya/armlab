import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Define a custom PyTorch Dataset
class MINCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Open image
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # Define the dataset root directory
    DATASET_ROOT = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\minc-2500"  # Change this to your dataset path

    # Load category names
    categories_path = os.path.join(DATASET_ROOT, "categories.txt")
    with open(categories_path, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    category_to_index = {cat: idx for idx, cat in enumerate(categories)}
    print(f"Number of categories: {len(categories)}")

    # Function to read image paths from text files
    def load_image_paths(label_file):
        image_paths = []
        labels = []
        with open(os.path.join(DATASET_ROOT, "labels", label_file), "r") as f:
            for line in f:
                rel_path = line.strip()  # Example: images/brick/brick_123.jpg
                category = rel_path.split("/")[1]  # Extract category name (e.g., "brick")
                full_path = os.path.join(DATASET_ROOT, rel_path)
                if os.path.exists(full_path):  # Ensure the image file exists
                    image_paths.append(full_path)
                    labels.append(category_to_index[category])
        return image_paths, labels

    # Load train, validation, and test sets
    train_images, train_labels = load_image_paths("train1.txt")  # Using split 1
    val_images, val_labels = load_image_paths("validate1.txt")
    test_images, test_labels = load_image_paths("test1.txt")
    print(f"Train size: {len(train_images)}, Val size: {len(val_images)}, Test size: {len(test_images)}")

    # Define transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

    # Create dataset instances
    train_dataset = MINCDataset(train_images, train_labels, transform=transform)
    val_dataset = MINCDataset(val_images, val_labels, transform=transform)
    test_dataset = MINCDataset(test_images, test_labels, transform=transform)
    print("Datasets created.")

    # Detect available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use pin_memory=True for faster host-to-GPU transfers
    num_workers = 4 if torch.cuda.is_available() else 0  # Multiprocessing can be problematic on Windows

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("Data loaders created.")

    # Load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)
    num_classes = len(categories)  # Number of classes in MINC-2500

    # Replace the final fully connected layer with a new one for MINC-2500
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to GPU
    model.to(device)
    print("Model moved to device.")

    criterion = nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # You can adjust learning rate
    print("Model created.")

    num_epochs = 10  # Change this as needed

    print("Training started.")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation
        model.eval()  # Switch model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training complete.")
    torch.save(model.state_dict(), "minc_resnet50.pth")

if __name__ == "__main__":
    main()

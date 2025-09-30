import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 1: Preparing the data
# Converting the images to tensor
transform = transforms.ToTensor()


# Downloading MNIST
train_datasets = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_datasets = datasets.MNIST(root="./data",train=False, download=True, transform=transform)


# Data loaders
train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=64, shuffle=False)


# Step 2: Defining the Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
        
        
    def forward(self,x):
        x = x.view(-1, 28*28)   # Flatten
        x = F.relu(self.fc1(x)) # Hidden layer 1
        x = F.relu(self.fc2(x)) # Hidden layer 2
        x = self.fc3(x)         # Output layer (logits)
        return x


# Initializing model and move to device
model =  SimpleNN().to(device)


# Step 3: Loss and Optimizer
# CrossEntropyLoss for classification
# SGD optimizer with momentum

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Step 4: Train Model
EPOCHS = 5
train_losses = []
test_accuracies = []


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    
    for data,targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
       
        
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
    
    
    # Evaluate 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1} Test Accuracy: {accuracy:2f} %")
    
    
# Step 5: Plot Trainign History
plt.figure(figsize=(12, 5))


plt.subplot(1,2,1)
plt.plot(train_losses, marker="*")
plt.title("Training Loss")
plt.xlabel('Epoch')
plt.ylabel("Loss")


plt.subplot(1,2,1)
plt.plot(test_accuracies, marker="*")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")


plt.show()


# Step 6: Visualize Predictions
# Take a batch from the test set
# Predict digits
# Display images with predicted and true labels
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


with torch.no_grad():
    example_data = example_data.to(device)
    outputs = model(example_data.view(example_data.shape[0], -1))
    
    
# Show first 6 images with predictions
fig = plt.figure(figsize=(15,10))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i].cpu().squeeze(), cmap="gray")
    plt.title(f"Pred: {outputs.data.max(1, keepdim=True)[1][i].item()} | True: {example_targets[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()

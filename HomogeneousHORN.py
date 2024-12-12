import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class sMNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data.view(data.size(0), -1).float()
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class HomogeneousHORNNetwork(nn.Module):
    def __init__(self, num_nodes, omega, gamma, alpha, sequence_length, output_classes=10):
        super(HomogeneousHORNNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.h = 1.0

        input_dim = 1
        self.W_ih = nn.Linear(input_dim, num_nodes, bias=True)
        self.W_hh = nn.Linear(num_nodes, num_nodes, bias=True)
        self.readout = nn.Linear(num_nodes, output_classes, bias=True)

        self.apply(self.initialize_weights)

        self.logged_x = None
        self.logged_v = None
        self.logged_a = None

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        device = input_sequence.device
        inputs = input_sequence.unsqueeze(-1)

        x = torch.zeros(batch_size, self.num_nodes, device=device)
        y = torch.zeros(batch_size, self.num_nodes, device=device)

        all_x = []
        all_y = []
        all_a = []

        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            I_rec = self.W_hh(y)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)

            accel = self.alpha * torch.tanh(A_input) - 2 * self.gamma * y - (self.omega**2) * x

            x = x + self.h * y
            y = y + self.h * accel

            all_x.append(x.clone())
            all_y.append(y.clone())
            all_a.append(accel.clone())

        self.logged_x = torch.stack(all_x, dim=0)
        self.logged_v = torch.stack(all_y, dim=0)
        self.logged_a = torch.stack(all_a, dim=0)

        output = self.readout(x)
        return output

def train_homogeneous_horn_network(horn_network, train_loader, test_loader, epochs=5, lr=0.001, save_data=True, save_dir='./results_homo'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horn_network = horn_network.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(horn_network.parameters(), lr=lr)

    train_accuracies = []
    test_accuracies = []
    losses = []

    classes = list(range(10))
    if save_data and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        horn_network.train()
        train_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = horn_network(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()

            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in horn_network.parameters() if p.grad is not None]), 2
            )
            gradient_norms.append(total_norm.item())

            torch.nn.utils.clip_grad_norm_(horn_network.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_grad_norm = np.mean(gradient_norms)
        avg_loss = train_loss
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Avg Grad Norm: {avg_grad_norm:.4f}")

        test_accuracy, test_preds, test_labels = evaluate(horn_network, test_loader, device)
        print(f"Test Accuracy After Epoch {epoch+1}: {test_accuracy:.2f}%")

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if (epoch+1) % 20 == 0:
            cm = compute_confusion_matrix(test_labels, test_preds, classes=len(classes))
            plot_confusion_matrix(cm, classes=classes, epoch=epoch+1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Over Epochs - Homogeneous HORN')
    plt.legend()
    plt.grid(True)
    plt.show()

    final_test_accuracy, _, _ = evaluate(horn_network, test_loader, device)
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")

    if save_data:
        results_path = os.path.join(save_dir, 'training_results.txt')
        with open(results_path, 'w') as f:
            f.write('Epoch\tTrain_Accuracy\tTest_Accuracy\tLoss\n')
            for i in range(epochs):
                f.write(f"{i+1}\t{train_accuracies[i]:.2f}\t{test_accuracies[i]:.2f}\t{losses[i]:.4f}\n")
        print(f"Training details saved to {results_path}")

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataset = sMNISTDataset(mnist_train.data, mnist_train.targets)
test_dataset = sMNISTDataset(mnist_test.data, mnist_test.targets)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)

num_nodes = 94
omega = 2 * np.pi / 28
gamma = 0.1
alpha = 0.5
sequence_length = 784

horn_network = HomogeneousHORNNetwork(num_nodes, omega, gamma, alpha, sequence_length)
train_homogeneous_horn_network(horn_network, train_loader, test_loader, epochs=100, lr=0.01, save_data=True, save_dir='./results_homo')

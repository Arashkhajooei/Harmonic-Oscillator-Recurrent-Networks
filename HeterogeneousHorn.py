import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

class HeterogeneousHORNNetwork(nn.Module):
    def __init__(self, num_nodes, base_omega, base_gamma, base_alpha, sequence_length, output_classes=10, hetero_std=0.01):
        super(HeterogeneousHORNNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.h = 1.0
        self.omega = nn.Parameter(torch.ones(num_nodes) * base_omega, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(num_nodes) * base_gamma, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(num_nodes) * base_alpha, requires_grad=True)

        with torch.no_grad():
            self.omega += hetero_std * torch.randn(num_nodes)
            self.gamma += hetero_std * torch.randn(num_nodes)
            self.alpha += hetero_std * torch.randn(num_nodes)

        self.W_ih = nn.Linear(1, num_nodes, bias=True)
        self.W_hh = nn.Linear(num_nodes, num_nodes, bias=True)
        self.readout = nn.Linear(num_nodes, output_classes, bias=True)
        self.apply(self.initialize_weights)

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

        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            I_rec = self.W_hh(y)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)

            alpha = self.alpha.unsqueeze(0)
            gamma = self.gamma.unsqueeze(0)
            omega = self.omega.unsqueeze(0)

            accel = alpha * torch.tanh(A_input) - 2 * gamma * y - (omega ** 2) * x
            x = x + self.h * y
            y = y + self.h * accel

        output = self.readout(x)
        return output

def evaluate(horn_network, loader, device):
    horn_network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = horn_network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def train_heterogeneous_horn_network(horn_network, train_loader, test_loader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horn_network = horn_network.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(horn_network.parameters(), lr=lr)

    train_accuracies = []
    test_accuracies = []
    losses = []

    for epoch in range(epochs):
        horn_network.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = horn_network(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(horn_network.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * correct / total
        test_accuracy = evaluate(horn_network, test_loader, device)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        losses.append(train_loss)

    plt.figure()
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Heterogeneous HORN Training')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_dataset = sMNISTDataset(mnist_train.data, mnist_train.targets)
    test_dataset = sMNISTDataset(mnist_test.data, mnist_test.targets)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)

    num_nodes = 94
    base_omega = 2 * np.pi / 28
    base_gamma = 0.1
    base_alpha = 0.5
    sequence_length = 784

    horn_network = HeterogeneousHORNNetwork(num_nodes, base_omega, base_gamma, base_alpha, sequence_length)
    train_heterogeneous_horn_network(horn_network, train_loader, test_loader, epochs=100, lr=0.001)

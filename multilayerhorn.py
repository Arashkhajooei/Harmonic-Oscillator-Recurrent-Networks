import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataset = sMNISTDataset(mnist_train.data, mnist_train.targets)
test_dataset = sMNISTDataset(mnist_test.data, mnist_test.targets)

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, pin_memory=True)

sequence_length = 784

class BaseHORN(nn.Module):
    def __init__(self, num_nodes, omega, gamma, alpha, sequence_length, output_classes=10):
        super(BaseHORN, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.h = 1.0

        self.W_ih = nn.Linear(1, num_nodes, bias=True)
        self.W_hh = nn.Linear(num_nodes, num_nodes, bias=True)
        self.readout = nn.Linear(num_nodes, output_classes, bias=True)

        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward_single_layer(self, input_sequence, omega, gamma, alpha):
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
            accel = alpha * torch.tanh(A_input) - 2*gamma*y - (omega**2)*x
            x = x + self.h * y
            y = y + self.h * accel
        return x

    def forward(self, input_sequence):
        x = self.forward_single_layer(input_sequence, self.omega, self.gamma, self.alpha)
        output = self.readout(x)
        return output

class HomogeneousHORN(BaseHORN):
    pass

class HeterogeneousHORN(BaseHORN):
    def __init__(self, num_nodes, base_omega, base_gamma, base_alpha, sequence_length, hetero_std=0.01, output_classes=10):
        super(BaseHORN, self).__init__()
        nn.Module.__init__(self)
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.h = 1.0

        self.omega = nn.Parameter(torch.ones(num_nodes)*base_omega, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(num_nodes)*base_gamma, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(num_nodes)*base_alpha, requires_grad=True)

        with torch.no_grad():
            self.omega += hetero_std * torch.randn(num_nodes)
            self.gamma += hetero_std * torch.randn(num_nodes)
            self.alpha += hetero_std * torch.randn(num_nodes)

        self.W_ih = nn.Linear(1, num_nodes, bias=True)
        self.W_hh = nn.Linear(num_nodes, num_nodes, bias=True)
        self.readout = nn.Linear(num_nodes, output_classes, bias=True)

        self.apply(self.initialize_weights)

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        device = input_sequence.device
        inputs = input_sequence.unsqueeze(-1)

        x = torch.zeros(batch_size, self.num_nodes, device=device)
        y = torch.zeros(batch_size, self.num_nodes, device=device)

        alpha = self.alpha.unsqueeze(0)
        gamma = self.gamma.unsqueeze(0)
        omega = self.omega.unsqueeze(0)

        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            I_rec = self.W_hh(y)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)
            accel = alpha * torch.tanh(A_input) - 2*gamma*y - (omega**2)*x
            x = x + self.h * y
            y = y + self.h * accel

        output = self.readout(x)
        return output

class MultiLayerHORN(nn.Module):
    def __init__(self, num_nodes_layer1, num_nodes_layer2, omega, gamma, alpha, sequence_length, output_classes=10):
        super(MultiLayerHORN, self).__init__()
        self.num_nodes_l1 = num_nodes_layer1
        self.num_nodes_l2 = num_nodes_layer2
        self.sequence_length = sequence_length
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.h = 1.0

        self.W_ih_1 = nn.Linear(1, num_nodes_layer1)
        self.W_hh_1 = nn.Linear(num_nodes_layer1, num_nodes_layer1)
        self.W_ih_2 = nn.Linear(num_nodes_layer1, num_nodes_layer2)
        self.W_hh_2 = nn.Linear(num_nodes_layer2, num_nodes_layer2)
        self.readout = nn.Linear(num_nodes_layer2, output_classes)

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
        x1 = torch.zeros(batch_size, self.num_nodes_l1, device=device)
        y1 = torch.zeros(batch_size, self.num_nodes_l1, device=device)
        x2 = torch.zeros(batch_size, self.num_nodes_l2, device=device)
        y2 = torch.zeros(batch_size, self.num_nodes_l2, device=device)
        omega = self.omega
        gamma = self.gamma
        alpha = self.alpha
        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext_1 = self.W_ih_1(s_t)
            I_rec_1 = self.W_hh_1(y1)
            A_input_1 = (I_ext_1 + I_rec_1) / np.sqrt(self.num_nodes_l1)
            accel_1 = alpha * torch.tanh(A_input_1) - 2*gamma*y1 - (omega**2)*x1
            x1 = x1 + self.h * y1
            y1 = y1 + self.h * accel_1

            I_ext_2 = self.W_ih_2(x1)
            I_rec_2 = self.W_hh_2(y2)
            A_input_2 = (I_ext_2 + I_rec_2) / np.sqrt(self.num_nodes_l2)
            accel_2 = alpha * torch.tanh(A_input_2) - 2*gamma*y2 - (omega**2)*x2
            x2 = x2 + self.h * y2
            y2 = y2 + self.h * accel_2
        output = self.readout(x2)
        return output
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def train_model(model, train_loader, test_loader, epochs, lr, device, arch_name, results):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_accuracies = []
    test_accuracies = []
    losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"{arch_name} Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100.0 * correct / total
        test_accuracy = evaluate(model, test_loader, device)
        print(f"{arch_name} Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, Test Acc={test_accuracy:.2f}%")
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        losses.append(train_loss)
        results.append((arch_name, epoch+1, train_accuracy, test_accuracy, train_loss))
    return train_accuracies, test_accuracies, losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    lr = 0.001
    base_omega = 2 * np.pi / 28
    base_gamma = 0.1
    base_alpha = 0.5
    architectures = []
    homo16 = HomogeneousHORN(num_nodes=16, omega=base_omega, gamma=base_gamma, alpha=base_alpha, sequence_length=sequence_length)
    architectures.append(("HOMO_16", homo16))
    homo32 = HomogeneousHORN(num_nodes=32, omega=base_omega, gamma=base_gamma, alpha=base_alpha, sequence_length=sequence_length)
    architectures.append(("HOMO_32", homo32))
    multi_32_32 = MultiLayerHORN(num_nodes_layer1=32, num_nodes_layer2=32, omega=base_omega, gamma=base_gamma, alpha=base_alpha, sequence_length=sequence_length)
    architectures.append(("HOMO_MULTI_32_32", multi_32_32))
    hete32 = HeterogeneousHORN(num_nodes=32, base_omega=base_omega, base_gamma=base_gamma, base_alpha=base_alpha, sequence_length=sequence_length, hetero_std=0.01)
    architectures.append(("HETE_32", hete32))
    all_results = []
    all_train_acc = {}
    all_test_acc = {}
    for arch_name, model in architectures:
        train_acc, test_acc, losses = train_model(model, train_loader, test_loader, epochs, lr, device, arch_name, all_results)
        all_train_acc[arch_name] = train_acc
        all_test_acc[arch_name] = test_acc
    plt.figure(figsize=(10,6))
    for arch_name in all_train_acc:
        plt.plot(range(1, epochs+1), all_train_acc[arch_name], label=f'{arch_name} Train')
        plt.plot(range(1, epochs+1), all_test_acc[arch_name], label=f'{arch_name} Test', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy for Different Architectures')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_architectures_accuracy.png')
    plt.show()
    with open('all_architecture_results.txt', 'w') as f:
        f.write("Architecture\tEpoch\tTrainAcc\tTestAcc\tLoss\n")
        for r in all_results:
            arch_name, ep, tr_acc, te_acc, loss = r
            f.write(f"{arch_name}\t{ep}\t{tr_acc:.2f}\t{te_acc:.2f}\t{loss:.4f}\n")
    print("Results saved to all_architecture_results.txt and plot saved to all_architectures_accuracy.png")

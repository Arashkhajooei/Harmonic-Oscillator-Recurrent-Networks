import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class sMNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data.view(data.size(0), -1).float()
        self.targets = targets.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class OrientedLineDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=28, orientations=[0, 45, 90, 135]):
        self.num_samples = num_samples
        self.image_size = image_size
        self.orientations = orientations
        self.data = []
        self.labels = []
        self.generate_data()

    def generate_data(self):
        for _ in range(self.num_samples):
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            orientation = np.random.choice(self.orientations)
            cx, cy = self.image_size // 2, self.image_size // 2
            length = self.image_size // 2
            rad = np.deg2rad(orientation)
            x1 = int(cx - length * np.cos(rad))
            y1 = int(cy - length * np.sin(rad))
            x2 = int(cx + length * np.cos(rad))
            y2 = int(cy + length * np.sin(rad))
            xs = np.linspace(x1, x2, length * 2).astype(int)
            ys = np.linspace(y1, y2, length * 2).astype(int)
            xs = np.clip(xs, 0, self.image_size - 1)
            ys = np.clip(ys, 0, self.image_size - 1)
            img[ys, xs] = 1.0
            self.data.append(img.flatten())
            self.labels.append(self.orientations.index(orientation))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataset_sMNIST = sMNISTDataset(mnist_train.data, mnist_train.targets)
test_dataset_sMNIST = sMNISTDataset(mnist_test.data, mnist_test.targets)

oriented_train_dataset = OrientedLineDataset(num_samples=10000)
oriented_test_dataset = OrientedLineDataset(num_samples=2000)

train_loader_orient = DataLoader(oriented_train_dataset, batch_size=4096, shuffle=True, pin_memory=True)
test_loader_orient = DataLoader(oriented_test_dataset, batch_size=4096, shuffle=False, pin_memory=True)

train_loader_sMNIST = DataLoader(train_dataset_sMNIST, batch_size=4096, shuffle=True, pin_memory=True)
test_loader_sMNIST = DataLoader(test_dataset_sMNIST, batch_size=4096, shuffle=False, pin_memory=True)

class HeterogeneousHORN(nn.Module):
    def __init__(self, num_nodes, base_omega, base_gamma, base_alpha, sequence_length, output_classes=10, hetero_std=0.01):
        super(HeterogeneousHORN, self).__init__()
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

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.W_ih.weight)
        nn.init.zeros_(self.W_ih.bias)
        nn.init.xavier_uniform_(self.W_hh.weight)
        nn.init.zeros_(self.W_hh.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, input_sequence):
        B = input_sequence.size(0)
        device = input_sequence.device
        inputs = input_sequence.unsqueeze(-1)

        x = torch.zeros(B, self.num_nodes, device=device)
        y = torch.zeros(B, self.num_nodes, device=device)

        alpha = self.alpha.unsqueeze(0)
        gamma = self.gamma.unsqueeze(0)
        omega = self.omega.unsqueeze(0)

        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            I_rec = self.W_hh(y)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)

            accel = alpha * torch.tanh(A_input) - 2 * gamma * y - (omega ** 2) * x

            x = x + self.h * y
            y = y + self.h * accel

        output = self.readout(x)
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

def train_model(model, train_loader, test_loader, epochs, lr, device, freeze_layers=False):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * correct / total
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, Test Acc={test_accuracy:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 10
    transfer_epochs = 50
    lr_pretrain = 0.001
    lr_transfer = 0.001
    base_omega = 2 * np.pi / 28
    base_gamma = 0.1
    base_alpha = 0.5
    num_nodes = 32
    sequence_length = 784
    hete32_pretrain = HeterogeneousHORN(
        num_nodes=num_nodes,
        base_omega=base_omega,
        base_gamma=base_gamma,
        base_alpha=base_alpha,
        sequence_length=sequence_length,
        output_classes=4
    )
    train_model(hete32_pretrain, train_loader_orient, test_loader_orient, pretrain_epochs, lr_pretrain, device)
    for param in hete32_pretrain.W_ih.parameters():
        param.requires_grad = False
    for param in hete32_pretrain.W_hh.parameters():
        param.requires_grad = False
    hete32_pretrain.readout = nn.Linear(num_nodes, 10, bias=True).to(device)
    nn.init.xavier_uniform_(hete32_pretrain.readout.weight)
    nn.init.zeros_(hete32_pretrain.readout.bias)
    train_model(hete32_pretrain, train_loader_sMNIST, test_loader_sMNIST, transfer_epochs, lr_transfer, device)
    hete32_scratch = HeterogeneousHORN(
        num_nodes=num_nodes,
        base_omega=base_omega,
        base_gamma=base_gamma,
        base_alpha=base_alpha,
        sequence_length=sequence_length,
        output_classes=10
    )
    train_model(hete32_scratch, train_loader_sMNIST, test_loader_sMNIST, transfer_epochs, lr_transfer, device)

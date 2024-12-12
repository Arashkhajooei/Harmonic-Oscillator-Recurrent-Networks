import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class OrientedLineDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=28, orientations=[0,45,90,135]):
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
            num_points = length * 2
            xs = np.linspace(x1, x2, num_points).astype(int)
            ys = np.linspace(y1, y2, num_points).astype(int)
            xs = np.clip(xs, 0, self.image_size - 1)
            ys = np.clip(ys, 0, self.image_size - 1)
            img[ys, xs] = 1.0
            self.data.append(img.flatten())
            self.labels.append(self.orientations.index(orientation))
        self.data = np.stack(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataset_sMNIST = sMNISTDataset(mnist_train.data, mnist_train.targets)
test_dataset_sMNIST = sMNISTDataset(mnist_test.data, mnist_test.targets)

oriented_train_dataset = OrientedLineDataset(num_samples=10000)
oriented_test_dataset = OrientedLineDataset(num_samples=2000)

train_loader_orient = DataLoader(oriented_train_dataset, batch_size=1024, shuffle=True, pin_memory=True)
test_loader_orient = DataLoader(oriented_test_dataset, batch_size=1024, shuffle=False, pin_memory=True)

train_loader_sMNIST = DataLoader(train_dataset_sMNIST, batch_size=1024, shuffle=True, pin_memory=True)
test_loader_sMNIST = DataLoader(test_dataset_sMNIST, batch_size=1024, shuffle=False, pin_memory=True)

sequence_length = 784

class DelayedHORN(nn.Module):
    def __init__(self, num_nodes, omega, gamma, alpha, sequence_length, delay=5, output_classes=10):
        super(DelayedHORN, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.h = 1.0
        self.delay = delay
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
        past_ys = []
        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            if t - self.delay >= 0:
                y_delayed = past_ys[t - self.delay]
            else:
                y_delayed = torch.zeros_like(y)
            I_rec = self.W_hh(y_delayed)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)
            accel = self.alpha * torch.tanh(A_input) - 2 * self.gamma * y - (self.omega ** 2) * x
            x = x + self.h * y
            y = y + self.h * accel
            past_ys.append(y.clone())
        output = self.readout(x)
        return output

class HomogeneousDelayedHORN(nn.Module):
    def __init__(self, num_nodes, omega, gamma, alpha, sequence_length, delay=5, output_classes=10):
        super(HomogeneousDelayedHORN, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.h = 1.0
        self.delay = delay
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
        alpha = torch.ones(1, self.num_nodes, device=device) * self.alpha
        gamma = torch.ones(1, self.num_nodes, device=device) * self.gamma
        omega = torch.ones(1, self.num_nodes, device=device) * self.omega
        past_ys = []
        for t in range(self.sequence_length):
            s_t = inputs[:, t]
            I_ext = self.W_ih(s_t)
            if t - self.delay >= 0:
                y_delayed = past_ys[t - self.delay]
            else:
                y_delayed = torch.zeros_like(y)
            I_rec = self.W_hh(y_delayed)
            A_input = (I_ext + I_rec) / np.sqrt(self.num_nodes)
            accel = alpha * torch.tanh(A_input) - 2 * gamma * y - (omega ** 2) * x
            x = x + self.h * y
            y = y + self.h * accel
            past_ys.append(y.clone())
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

def train_model(model, train_loader, test_loader, epochs, lr, device, arch_name="Model"):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_accuracies = []
    test_accuracies = []
    losses = []
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"{arch_name} Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        all_results.append((arch_name, epoch+1, train_accuracy, test_accuracy, train_loss))
    return train_accuracies, test_accuracies, losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_epochs = 100
    lr_train = 0.001
    base_omega = 2 * np.pi / 28
    base_gamma = 0.1
    base_alpha = 0.5
    delay = 5
    architectures_scratch = []
    homo16_delayed_scratch = HomogeneousDelayedHORN(num_nodes=16, omega=base_omega, gamma=base_gamma, alpha=base_alpha,
                                                   sequence_length=sequence_length, delay=delay, output_classes=10)
    architectures_scratch.append(("HOMO_16_Delayed_Scratch", homo16_delayed_scratch))
    homo32_delayed_scratch = HomogeneousDelayedHORN(num_nodes=32, omega=base_omega, gamma=base_gamma, alpha=base_alpha,
                                                   sequence_length=sequence_length, delay=delay, output_classes=10)
    architectures_scratch.append(("HOMO_32_Delayed_Scratch", homo32_delayed_scratch))
    hete32_delayed_scratch = DelayedHORN(num_nodes=32, omega=base_omega, gamma=base_gamma, alpha=base_alpha,
                                         sequence_length=sequence_length, delay=delay, output_classes=10)
    architectures_scratch.append(("HETE_32_Delayed_Scratch", hete32_delayed_scratch))
    all_results = []
    all_train_acc = {}
    all_test_acc = {}
    for arch_name, model in architectures_scratch:
        model.to(device)
        train_acc, test_acc, losses = train_model(model, train_loader_sMNIST, test_loader_sMNIST,
                                                epochs=train_epochs, lr=lr_train, device=device,
                                                arch_name=arch_name)
        all_train_acc[arch_name] = train_acc
        all_test_acc[arch_name] = test_acc
    plt.figure(figsize=(14, 8))
    for arch_name in all_train_acc:
        plt.plot(range(1, train_epochs +1), all_train_acc[arch_name], label=f'{arch_name} Train')
        plt.plot(range(1, train_epochs +1), all_test_acc[arch_name], label=f'{arch_name} Test', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy for Different HORN Architectures with Coupling Delays')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_architectures_accuracy_coupling_delays.png')
    plt.show()
    with open('all_architecture_results_coupling_delays.txt', 'w') as f:
        f.write("Architecture\tEpoch\tTrainAcc\tTestAcc\tLoss\n")
        for r in all_results:
            arch_name, ep, tr_acc, te_acc, loss = r
            f.write(f"{arch_name}\t{ep}\t{tr_acc:.2f}\t{te_acc:.2f}\t{loss:.4f}\n")
    print("Results saved to all_architecture_results_coupling_delays.txt and plot saved to all_architectures_accuracy_coupling_delays.png")

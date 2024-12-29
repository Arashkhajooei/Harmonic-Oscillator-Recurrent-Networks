import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class SequentialMNISTDataset(datasets.MNIST):
    """
    Custom Dataset class to process MNIST images row-wise.
    Each image is treated as a sequence of 28 rows, each row being a timestep.
    """
    def __getitem__(self, index):
        # Get the original MNIST image and label
        img, label = super(SequentialMNISTDataset, self).__getitem__(index)
        # Reshape the image into a sequence of 28 rows (timesteps)
        img_seq = img.squeeze(0).permute(1, 0)  # Shape: (28, 28) - each row is a timestep
        return img_seq, label

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Initialize datasets
train_dataset = SequentialMNISTDataset(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = SequentialMNISTDataset(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# Define DataLoaders
batch_size = 4096
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


class HybridReservoirModel(nn.Module):
    def __init__(self, num_nodes=512, output_dim=10, spectral_radius=0.9):
        super(HybridReservoirModel, self).__init__()
        self.num_nodes = num_nodes

        # Initialize W_hh with spectral radius normalization
        W_hh = torch.randn(num_nodes, num_nodes)
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W_hh)
            max_eigenvalue = torch.max(torch.abs(eigenvalues)).real
            W_hh = W_hh / max_eigenvalue * spectral_radius
        self.W_hh = nn.Parameter(W_hh, requires_grad=False)

        # Input and Output weights
        self.W_ih = nn.Parameter(torch.randn(28, num_nodes) * 0.01, requires_grad=True)  # 28 pixels per timestep
        self.W_out = nn.Parameter(torch.randn(num_nodes, output_dim) * 0.01, requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the reservoir.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        batch_size, seq_len, _ = x.size()
        r = torch.zeros(batch_size, self.num_nodes, device=x.device)  # Reservoir state

        for t in range(seq_len):
            x_t = x[:, t, :]  # Input at timestep t, shape: (batch_size, input_dim)
            in_signal = torch.matmul(x_t, self.W_ih)  # Shape: (batch_size, num_nodes)
            rec_signal = torch.matmul(r, self.W_hh)  # Shape: (batch_size, num_nodes)
            r = torch.tanh(in_signal + rec_signal)  # Update reservoir state

        logits = torch.matmul(r, self.W_out)  # Map reservoir state to output
        return logits


def evaluate(model, loader, device):
    """
    Evaluate the model's accuracy on a given dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)  # Shape: (batch_size, seq_len, input_dim)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

#####################################
# 4. Training
#####################################
def train_model(model, train_loader, test_loader, device, epochs, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    test_accuracies = []

    with tqdm(total=epochs, desc="Training Model") as pbar:
        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # Shape: (batch_size, seq_len, input_dim)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100.0 * correct / total
            test_acc = evaluate(model, test_loader, device)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            pbar.set_postfix({"Train Acc": f"{train_acc:.2f}%", "Test Acc": f"{test_acc:.2f}%"})
            pbar.update(1)

    return train_accuracies, test_accuracies


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the  reservoir model
    model = HybridReservoirModel(num_nodes=512, output_dim=10, spectral_radius=0.9).to(device)

    # Train and evaluate the model
    train_acc, test_acc = train_model(
        model, train_loader, test_loader, device, epochs=100, lr=5e-3
    )

    epochs_range = range(1, len(train_acc) + 1)  # Match the number of epochs dynamically
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_acc, label="Train Accuracy")
    plt.plot(epochs_range, test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Sequential MNIST: Reservoir Computing with BPTT")
    plt.legend()
    plt.grid(True)
    plt.show()


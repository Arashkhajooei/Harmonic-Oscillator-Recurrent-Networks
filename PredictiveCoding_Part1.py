import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class PredictiveCodingNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, iterations=50, lr=0.001):
        """
        Initialize the Predictive Coding Network.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer.
            iterations (int): Number of iterations for the predictive coding updates.
            lr (float): Internal learning rate for updating hidden states and outputs.
        """
        super(PredictiveCodingNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.iterations = iterations
        self.lr = lr  # Internal learning rate

        # Recognition weights
        self.W_rec = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)  # [256, 784]
        self.W_out = nn.Parameter(torch.randn(output_size, hidden_size) * 0.1) # [10, 256]

        # Biases
        self.bias_rec = nn.Parameter(torch.zeros(hidden_size))  # [256]
        self.bias_out = nn.Parameter(torch.zeros(output_size))  # [10]
        self.bias_gen = nn.Parameter(torch.zeros(input_size))    # [784]

        # Initialize weights using Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x):
        """
        Forward pass of the Predictive Coding Network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 784].

        Returns:
            y (torch.Tensor): Final output tensor of shape [batch_size, 10].
            average_energy (torch.Tensor): Averaged energy over iterations.
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden states and output states
        h = torch.zeros(batch_size, self.hidden_size, device=device)  # [batch_size, 256]
        y = torch.zeros(batch_size, self.output_size, device=device)  # [batch_size, 10]

        total_energy = 0.0  # Initialize energy accumulation

        for _ in range(self.iterations):
            # Generative weights are the same as recognition weights: [256, 784]
            W_gen = self.W_rec  # [256, 784]

            # Predict input from hidden state
            x_pred = torch.sigmoid(torch.matmul(h, W_gen) + self.bias_gen)  # [batch_size, 784]

            # Compute prediction error for input
            e_x = x - x_pred  # [batch_size, 784]

            # Update hidden state based on input error
            h = h + self.lr * torch.matmul(e_x, self.W_rec.t())  # [batch_size, 256]

            # Predict output from hidden state
            y_pred = torch.matmul(h, self.W_out.t()) + self.bias_out  # [batch_size, 10]

            # Compute prediction error for output
            e_y = y - y_pred  # [batch_size, 10]

            # Update output state based on output error
            y = y + self.lr * e_y  # [batch_size, 10]

            # Calculate energy at this iteration
            energy = (e_x ** 2).sum(dim=1).mean() + (e_y ** 2).sum(dim=1).mean()  # Scalar
            total_energy += energy  # Accumulate energy

        # Normalize total energy by the number of iterations
        average_energy = total_energy / self.iterations  # Scalar

        # Return final output state and average energy
        return y, average_energy

def get_data_loaders(batch_size_train=2048, batch_size_test=1000):
    """
    Prepare the MNIST data loaders.

    Args:
        batch_size_train (int): Batch size for training.
        batch_size_test (int): Batch size for testing.

    Returns:
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch, max_grad_norm=5.0):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The predictive coding model.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        epoch (int): Current epoch number.
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        avg_loss (float): Average classification loss for the epoch.
        avg_energy (float): Average energy for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_energy = 0.0
    num_batches = len(train_loader)

    # Gradual energy weight increase with a slower schedule
    energy_weight = min(0.05, epoch / 200)  # Max energy_weight is 0.05 at epoch 200

    progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Epoch {epoch} [Train]', ncols=100)

    for batch_idx, (data, target) in progress_bar:
        data = data.view(-1, 784).to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output, energy = model(data)

        classification_loss = criterion(output, target)
        combined_loss = classification_loss + energy_weight * energy  # Weighted loss
        combined_loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += classification_loss.item()
        running_energy += energy.item()

        avg_loss = running_loss / (batch_idx + 1)
        avg_energy = running_energy / (batch_idx + 1)
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Energy': f'{avg_energy:.4f}'})

    return avg_loss, avg_energy

def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The predictive coding model.
        device (torch.device): Device to run the evaluation on.
        test_loader (DataLoader): Testing data loader.
        criterion (nn.Module): Loss function (CrossEntropyLoss).

    Returns:
        avg_test_loss (float): Average test loss.
        accuracy (float): Accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    num_batches = len(test_loader)

    progress_bar = tqdm(enumerate(test_loader), total=num_batches, desc='Evaluating', ncols=100)

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data = data.view(-1, 784).to(device, non_blocking=True)  # Flatten the images: [batch_size, 784]
            target = target.to(device, non_blocking=True)
            outputs, _ = model(data)  # Forward pass
            loss = criterion(outputs, target)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Predictions
            total += target.size(0)
            correct += (predicted == target).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / num_batches
    print(f'\nTest Loss: {avg_test_loss:.4f}, Accuracy on test set: {accuracy:.2f}%')
    return avg_test_loss, accuracy

def plot_sample_predictions(model, device, test_loader, num_samples=5):
    """
    Plot sample predictions from the test set.

    Args:
        model (nn.Module): The predictive coding model.
        device (torch.device): Device to run the predictions on.
        test_loader (DataLoader): Testing data loader.
        num_samples (int): Number of sample images to plot.
    """
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images = images.view(-1, 784).to(device)
    labels = labels.to(device)
    outputs, _ = model(images)
    _, predicted = torch.max(outputs.data, 1)

    plt.figure(figsize=(15, 3))
    for idx in range(num_samples):
        plt.subplot(1, num_samples, idx + 1)
        img = images[idx].cpu().view(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {labels[idx].item()}\nPred: {predicted[idx].item()}')
        plt.axis('off')
    plt.show()


train_losses, train_energies = [], []
test_losses, test_accuracies = [], []

def main():
    """
    Main function to train and evaluate the Predictive Coding Network.
    """
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Prepare data loaders
    train_loader, test_loader = get_data_loaders(batch_size_train=4096, batch_size_test=1000)

    # Initialize the model
    model = PredictiveCodingNet().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler: Reduce LR on plateau of test loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_epochs = 100


    for epoch in range(1, num_epochs + 1):
        print(f'\n--- Epoch {epoch} ---')
        train_loss, train_energy = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

        # Step the scheduler based on test loss
        scheduler.step(test_loss)

        # Record metrics
        train_losses.append(train_loss)
        train_energies.append(train_energy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Plot sample predictions
    plot_sample_predictions(model, device, test_loader)

    # Plot training and test loss over epochs
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_energies, label='Training Energy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.title('Energy per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot test accuracy over epochs
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10


# Data loading and preprocessing
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # Ensure outputs are between 0 and 1
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_the_model(latent_dim):
    # Instantiate model, loss, and optimizer
    model = Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return  avg_loss


def loss_vs_dim(latent_dims):
    """

    :param latent_dims: an array of dimensions [k,1]
    :return: None
    """
    latent_loss = np.zeros(latent_dims.shape)
    for i,latent_dim  in enumerate(latent_dims):
        latent_loss[i] = train_the_model(latent_dim)
    #Todo -3. plot the data
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(latent_dims, latent_loss, marker='o', linestyle='-', color='b')
    plt.title("Loss vs Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Show reconstructions
# show_reconstructions(model, test_loader)
latent_dims = np.arange(1, 28*28, 5)
loss_vs_dim(latent_dims)

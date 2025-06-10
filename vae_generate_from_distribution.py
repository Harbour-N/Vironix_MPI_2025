import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
latent_dim = 32

# FashionMNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data loading and preprocessing
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Latent space layers
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# VAE Loss function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# Instantiate model and optimizer
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        recon_images, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_kl_loss = total_kl_loss / len(train_loader.dataset)

    train_losses.append(avg_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Total Loss: {avg_loss:.4f}, '
          f'Recon Loss: {avg_recon_loss:.4f}, '
          f'KL Loss: {avg_kl_loss:.4f}')


# Function to extract latent representations and analyze by category
def analyze_latent_distributions(model, data_loader):
    model.eval()
    latent_means = {i: [] for i in range(10)}
    latent_logvars = {i: [] for i in range(10)}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            mu, logvar = model.encode(images)

            # Group by category
            for i in range(len(labels)):
                category = labels[i].item()
                latent_means[category].append(mu[i].cpu().numpy())
                latent_logvars[category].append(logvar[i].cpu().numpy())

    # Convert to numpy arrays and compute statistics
    category_stats = {}
    for category in range(10):
        means = np.array(latent_means[category])
        logvars = np.array(latent_logvars[category])

        category_stats[category] = {
            'mean_mu': np.mean(means, axis=0),
            'std_mu': np.std(means, axis=0),
            'mean_logvar': np.mean(logvars, axis=0),
            'std_logvar': np.std(logvars, axis=0),
            'mean_sigma': np.mean(np.exp(0.5 * logvars), axis=0),
            'std_sigma': np.std(np.exp(0.5 * logvars), axis=0)
        }

    return category_stats


# Analyze distributions for each category
print("\nAnalyzing latent space distributions by category...")
category_distributions = analyze_latent_distributions(model, test_loader)

# Display results
print("\n" + "=" * 80)
print("LATENT SPACE DISTRIBUTION ANALYSIS BY CATEGORY")
print("=" * 80)

for category in range(10):
    stats = category_distributions[category]
    print(f"\n{class_names[category]} (Category {category}):")
    print(f"  Mean μ: {stats['mean_mu'][:5]}... (showing first 5 dimensions)")
    print(f"  Std μ:  {stats['std_mu'][:5]}...")
    print(f"  Mean σ: {stats['mean_sigma'][:5]}...")
    print(f"  Std σ:  {stats['std_sigma'][:5]}...")
    print(f"  Mean log(σ²): {stats['mean_logvar'][:5]}...")


# Visualize latent space distributions
def plot_latent_distributions(category_stats, max_dims=8):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for dim in range(min(max_dims, latent_dim)):
        ax = axes[dim]

        # Plot mean values for each category
        means = [category_stats[cat]['mean_mu'][dim] for cat in range(10)]
        stds = [category_stats[cat]['std_mu'][dim] for cat in range(10)]

        x = range(10)
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(f'Latent Dimension {dim}')
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean μ')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i}' for i in range(10)], rotation=45)

    plt.tight_layout()
    plt.show()


# Visualize reconstructions
def show_reconstructions(model, data_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(data_loader))
        images = images[:num_images].to(device)
        labels = labels[:num_images]
        recon_images, mu, logvar = model(images)

    images = images.cpu().numpy()
    recon_images = recon_images.cpu().numpy()

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 1.5, 4.5))

    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{class_names[labels[i]]}', fontsize=8)

        # Reconstructed
        axes[1, i].imshow(recon_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

        # Show first few latent dimensions
        mu_sample = mu[i].cpu().numpy()
        logvar_sample = logvar[i].cpu().numpy()
        sigma_sample = np.exp(0.5 * logvar_sample)

        # Create a simple visualization of latent values
        latent_viz = np.zeros((4, 8))
        latent_viz[0, :] = mu_sample[:8]  # First 8 mu values
        latent_viz[1, :] = sigma_sample[:8]  # First 8 sigma values

        im = axes[2, i].imshow(latent_viz, cmap='coolwarm', aspect='auto')
        axes[2, i].axis('off')
        axes[2, i].set_title('μ & σ', fontsize=8)

    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    axes[2, 0].set_ylabel('Latent', fontsize=12)

    plt.tight_layout()
    plt.show()


# Generate sample images from latent space
def generate_samples(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)

    samples = samples.cpu().numpy()

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 1.5))
    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i + 1}', fontsize=8)

    plt.suptitle('Generated Samples from Latent Space')
    plt.tight_layout()
    plt.show()


# Show results
print("\nGenerating visualizations...")
plot_latent_distributions(category_distributions)
show_reconstructions(model, test_loader)
generate_samples(model)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Create a summary table of category distributions
import pandas as pd

summary_data = []
for category in range(10):
    stats = category_distributions[category]
    summary_data.append({
        'Category': class_names[category],
        'Mean_μ_L2_norm': np.linalg.norm(stats['mean_mu']),
        'Mean_σ_L2_norm': np.linalg.norm(stats['mean_sigma']),
        'Avg_μ': np.mean(stats['mean_mu']),
        'Avg_σ': np.mean(stats['mean_sigma']),
        'Std_μ': np.mean(stats['std_mu']),
        'Std_σ': np.mean(stats['std_sigma'])
    })

df_summary = pd.DataFrame(summary_data)
print("\n" + "=" * 80)
print("SUMMARY TABLE: Distribution Statistics by Category")
print("=" * 80)
print(df_summary.round(4))


def generate_from_distribution(model, category_distributions, category,
                               num_samples=5, device='cpu'):
    """
    Generate images using learned mu and sigma distributions for a specific category

    Args:
        model: Trained VAE model
        category_distributions: Dictionary with distribution stats from analyze_latent_distributions()
        category: Category index (0-9 for FashionMNIST)
        num_samples: Number of images to generate
        device: Device to run on

    Returns:
        Generated images tensor of shape (num_samples, 1, 28, 28)
    """

    if category not in category_distributions:
        raise ValueError(f"Category {category} not found in distributions")

    stats = category_distributions[category]
    mean_mu = torch.tensor(stats['mean_mu'], dtype=torch.float32, device=device)
    mean_sigma = torch.tensor(stats['mean_sigma'], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        # Sample from the learned distribution
        eps = torch.randn(num_samples, len(mean_mu), device=device)
        z = mean_mu.unsqueeze(0) + eps * mean_sigma.unsqueeze(0)

        # Generate images from latent codes
        generated_images = model.decode(z)

    return generated_images

sampled_boots= generate_from_distribution(model, category_distributions,
                                               category=9, num_samples=3,
                                              device=device)
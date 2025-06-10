

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

import utils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 100 # early parameter seeps of bottleneck dim suggest ~100 as an elbow.

# Data loading and preprocessing
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./MNIST_fashion', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),  # Ensure outputs are between 0 and 1
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#### TODO: save trained model?
# 

# Instantiate model, loss, and optimizer
model = Autoencoder().to(device)
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')



# Get the embedding of a single image
def get_embedding(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension: (1, 1, 28, 28)
        embedding = model.encoder(image_tensor)
    return embedding.squeeze().cpu()  # Remove batch dimension and move to CPU

# Take one image from test set
sample_image, label = test_dataset[0]

###
# Illustrate the embeddings:
# either PCA in original data,
# or first put it into the bottleneck of the embedding then projecting.

fig,ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)


# Embed then project
x = np.zeros((len(test_dataset), latent_dim))
y = np.zeros(len(test_dataset))

for i, (image, label) in enumerate(test_dataset):
    embedding = get_embedding(model, image)
    x[i] = embedding.numpy()
    y[i] = label  # Store the label

pca1 = PCA(n_components=2)
x_pca = pca1.fit_transform(x)

#

scatter = ax[0].scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)

pct_var_1 = sum(pca1.explained_variance_ratio_)*100

ax[0].set_title(f'Autoecoder -> PCA (pct var={pct_var_1:.1f}%)')
ax[0].set(xlabel='PCA Component 1', ylabel='PCA Component 2')
cax=fig.colorbar(scatter, ticks=range(10), label='Label')
#plt.clim(-0.5, 9.5) # todo: direct object-oriented version of manipulating colorbar limits?
ax[0].grid(True)

#
# projection with original data only.
npixels = np.prod(image.shape)
x2 = np.zeros((len(test_dataset), npixels))

for i, (image, label) in enumerate(test_dataset):
    #embedding = get_embedding(model, image)
    x2[i] = image.detach().numpy().flatten()
    y[i] = label  # Store the label

pca2 = PCA(n_components=2)
x2_pca = pca2.fit_transform(x2)

#

scatter2 = ax[1].scatter(x2_pca[:, 0], x2_pca[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
pct_var_2 = sum(pca2.explained_variance_ratio_)*100

ax[1].set_title(f'PCA only (pct var={pct_var_2:.1f}%)')
ax[1].set(xlabel='PCA Component 1', ylabel='PCA Component 2')
cax=fig.colorbar(scatter2, ticks=range(10), label='Label')
#plt.clim(-0.5, 9.5) # todo: direct object-oriented version of manipulating colorbar limits?
ax[1].grid(True)

suffix = utils.tstamp()

fig.savefig(f'output/fashion_mnist_autoencoder_test_{suffix}.png', bbox_inches='tight')
fig.savefig(f'output/fashion_mnist_autoencoder_test_{suffix}.pdf', bbox_inches='tight')

#########



# Visualize reconstructions
def show_reconstructions(model, data_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images[:num_images].to(device)
        outputs = model(images)

    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    plt.tight_layout()
    #plt.show()
    plt.show(block=False)

# Show reconstructions
#show_reconstructions(model, test_loader)


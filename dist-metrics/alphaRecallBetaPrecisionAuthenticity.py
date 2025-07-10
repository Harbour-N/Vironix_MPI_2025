# === Acknowledgements ===
# This code was created with the help of Chatgpt
# Most of the code was copied verbatim from the code of the authors of
# the original paper (https://arxiv.org/abs/2102.08921) that proposed the metrics.
# The code is from the following repos
# - github.com/ahmedmalaa/evaluating-generative-models
# - https://github.com/vanderschaarlab/evaluating-generative-models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

import logging
import scipy

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from torch.autograd import Variable

device = 'cpu' # matrices are too big for gpu


# === Supporting Functions ===

# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

torch.manual_seed(1)

# Global variables

ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(),
                   "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(),
                   "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(),
                   "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(),
                   "SELU": torch.nn.SELU(),
                   "GLU": torch.nn.GLU(),
                   "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(),
                   "Softplus": torch.nn.Softplus()}


def build_network(network_name, params):

    if network_name=="feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params):

    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules          = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(torch.nn.Linear(params["input_dim"], params["num_hidden"],bias=False))
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(torch.nn.Linear(params["num_hidden"], params["num_hidden"],
                                       bias=False))
        modules.append(ACTIVATION_DICT[params["activation"]])


    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"],bias=False))

    _architecture    = nn.Sequential(*modules)

    return _architecture

# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

  -----------------------------------------
  One-class representations
  -----------------------------------------

"""

# One-class loss functions
# ------------------------


def OneClassLoss(outputs, c):

    dist   = torch.sum((outputs - c) ** 2, dim=1)
    loss   = torch.mean(dist)

    return loss


def SoftBoundaryLoss(outputs, R, c, nu):

    dist   = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R ** 2
    loss   = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    scores = dist
    loss   = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss


LossFns    = dict({"OneClass": OneClassLoss, "SoftBoundary": SoftBoundaryLoss})

# Base network
# ---------------------

class BaseNet(nn.Module):

    """Base class for all neural networks."""

    def __init__(self):

        super().__init__()

        self.logger  = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):

        """Forward pass logic

        :return: Network output
        """
        raise NotImplementedError

    def summary(self):

        """Network summary."""

        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params         = sum([np.prod(p.size()) for p in net_parameters])

        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


def get_radius(dist:torch.Tensor, nu:float):

    """Optimally solve for radius R via the (1-nu)-quantile of distances."""

    return np.quantile(np.sqrt(dist.clone().data.float().numpy()), 1 - nu)

class OneClassLayer(BaseNet):

    def __init__(self, params=None, hyperparams=None):

        super().__init__()

        # set all representation parameters - remove these lines

        self.rep_dim        = params["rep_dim"]
        self.input_dim      = params["input_dim"]
        self.num_layers     = params["num_layers"]
        self.num_hidden     = params["num_hidden"]
        self.activation     = params["activation"]
        self.dropout_prob   = params["dropout_prob"]
        self.dropout_active = params["dropout_active"]
        self.loss_type      = params["LossFn"]
        self.train_prop     = params['train_prop']
        self.learningRate   = params['lr']
        self.epochs         = params['epochs']
        self.warm_up_epochs = params['warm_up_epochs']
        self.weight_decay   = params['weight_decay']
        if torch.cuda.is_available():
            self.device     = torch.device('cuda') # Make this an option
        else:
            self.device     = torch.device('cpu')
        # set up the network

        self.model          = build_network(network_name="feedforward", params=params).to(self.device)

        # create the loss function

        self.c              = hyperparams["center"].to(self.device)
        self.R              = hyperparams["Radius"]
        self.nu             = hyperparams["nu"]

        self.loss_fn        = LossFns[self.loss_type]


    def forward(self, x):

        x                   = self.model(x)

        return x


    def fit(self, x_train, verbosity=True):


        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay = self.weight_decay)
        self.X              = torch.tensor(x_train.reshape((-1, self.input_dim))).float()

        if self.train_prop != 1:
            x_train, x_val = x_train[:int(self.train_prop*len(x_train))], x_train[int(self.train_prop*len(x_train)):]
            inputs_val = Variable(x_val)
            #inputs_val = Variable(torch.from_numpy(x_val).to(self.device)).float()

        self.losses         = []
        self.loss_vals       = []


        for epoch in range(self.epochs):

            # Converting inputs and labels to Variable

            inputs = Variable(x_train)#Variable(torch.from_numpy(x_train)).to(self.device).float()

            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output

            if self.loss_type=="SoftBoundary":

                self.loss = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu)

            elif self.loss_type=="OneClass":

                self.loss = self.loss_fn(outputs=outputs, c=self.c)


            #self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())

            # update parameters
            self.optimizer.step()

            if (epoch >= self.warm_up_epochs) and (self.loss_type=="SoftBoundary"):

                dist   = torch.sum((outputs - self.c) ** 2, dim=1)
                self.R = torch.tensor(get_radius(dist, self.nu)) ## THIS WAS COMMENTED OUT

            if self.train_prop != 1.0:
                with torch.no_grad():

                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)

                    # get loss for the predicted output

                    if self.loss_type=="SoftBoundary":

                        loss_val = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu)

                    elif self.loss_type=="OneClass":

                        loss_val = self.loss_fn(outputs=outputs, c=self.c).detach.cpu().numpy()

                    self.loss_vals.append(loss_val)




            if verbosity:
                if self.train_prop == 1:
                    print('epoch {}, loss {}'.format(epoch, self.loss.item()))
                else:
                    print('epoch {:4}, train loss {:.4e}, val loss {:.4e}'.format(epoch, self.loss.item(),loss_val))


# === Our Phi FUnction ===

# initialize the NN
params = {"rep_dim": 2,
          "input_dim": 2,
          "num_layers": 2,
          "num_hidden": 2,
          "activation": 'ReLU',
          "dropout_prob": 0.3,
          "dropout_active": True,
          "LossFn": "SoftBoundary",
          "train_prop": 0.8,
          'lr': 0.005,
          'epochs': 1000,
          'warm_up_epochs': 100,
          'weight_decay':1e-3}
hyperparams = {'center':torch.tensor([0, 0]),
               'Radius': 1.0,#torch.tensor([1.0]),
               'nu': 1.0}
ourPhi = OneClassLayer(params=params, hyperparams=hyperparams)

# train the the Phi
latent_dim = 2
nx = 150
x_train = torch.randn(nx, latent_dim).to('cpu') + torch.tensor([1, 1])
#x_train = sample_from_gaussian_mixture_torch(means, covariances, weights, ny)
ourPhi.fit(x_train=x_train, verbosity=False)

# plot the output
with torch.no_grad():
  print(ourPhi.R)
  test_outs = ourPhi(x_train)
  fig, ax = plt.subplots()
  ax.scatter(x_train[:,0], x_train[:, 1], alpha=1, s=5)
  ax.scatter(test_outs[:,0], test_outs[:, 1], alpha=1, s=3)
  plt.show()


# === Implement the Metrics ===
# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

  -----------------------------------------
  Metrics implementation
  -----------------------------------------

"""

def compute_alpha_precision(real_data, synthetic_data, emb_center):


    emb_center = torch.tensor(emb_center, device=device)

    n_steps = 30
    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)


    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)

    synth_center          = torch.mean(synthetic_data, axis=0) #torch.tensor(np.mean(synthetic_data, axis=0)).float()

    alpha_precision_curve = []
    beta_coverage_curve   = []

    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))


    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _       = nbrs_real.kneighbors(real_data)

    nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)

    # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
    real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
    real_to_synth_args    = real_to_synth_args.squeeze()

    real_synth_closest    = synthetic_data[real_to_synth_args]

    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
    closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)



    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())

        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)


    # See which one is bigger

    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen.numpy())

    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])

    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity


# === Test implementation on two distributions ===
# create real and synthetic data
nx = 150
ny = 150
latent_dim = 2
X = torch.randn(nx, latent_dim).to('cpu')

# sample from a mixture of gaussians
def sample_from_gaussian_mixture(means, covariances, weights, n_samples, device='cpu'):
    """
    Sample from a mixture of 2D Gaussian distributions using PyTorch.
    """
    assert len(means) == len(covariances) == len(weights), "Inconsistent component lengths"
    assert torch.isclose(torch.tensor(weights).sum(), torch.tensor(1.0)), "Weights must sum to 1"

    n_components = len(weights)
    component_choices = torch.multinomial(torch.tensor(weights), n_samples, replacement=True)
    samples = torch.zeros((n_samples, 2), device=device)

    for i in range(n_components):
        idx = (component_choices == i).nonzero(as_tuple=True)[0]
        count = idx.shape[0]
        if count > 0:
            dist = torch.distributions.MultivariateNormal(
                loc=means[i].to(device),
                covariance_matrix=covariances[i].to(device)
            )
            samples_i = dist.sample((count,))
            samples[idx] = samples_i

    return samples.cpu()

# Define mixture of 5 components
means = [
    torch.tensor([0.0, 0.0]),
    torch.tensor([10.0, 10.0]),
    torch.tensor([-5.0, 5.0]),
    torch.tensor([5.0, -5.0]),
    torch.tensor([-5., -5.0])
]

covariances = [
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    torch.tensor([[0.1, 0.0], [0.0, 0.1]]),
    torch.tensor([[0.6, 0.0], [0.0, 0.6]]),
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    torch.tensor([[0.1, 0.0], [0.0, 0.1]])
]

weights = [0.51, 0.015, 0.2, 0.2, 0.075]

Y = sample_from_gaussian_mixture(means, covariances, weights, ny)
center = torch.tensor([0, 0])

alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity = compute_alpha_precision(X, Y, center)

# make a plot
fig, ax = plt.subplots()

ax.plot(alphas, alpha_precision_curve)
ax.plot(alphas, alphas)


# === Test a GAN model for Phi ===
# train Phi using a GAN Model
def sample_unit_ball(n, num_samples=1, radius=1.0, device='cpu'):
    """
    Samples uniformly from the n-dimensional ball of given radius.

    Parameters:
        n (int): Dimension of the space.
        num_samples (int): Number of samples to draw.
        radius (float): Radius of the ball.
        device (str): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: Tensor of shape (num_samples, n) with samples in the ball.
    """
    # Step 1: Sample from standard normal and normalize to unit vectors
    direction = torch.randn(num_samples, n, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True)

    # Step 2: Sample radius scaling using U^(1/n)
    scales = torch.rand(num_samples, device=device) ** (1.0 / n)

    # Step 3: Scale directions and apply ball radius
    samples = radius * direction * scales.unsqueeze(1)
    return samples


# Generator: maps 2D noise -> 2D output
class Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=256, output_dim=2):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# Discriminator: maps 2D data -> scalar (real/fake)
class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def sample_real_data(batch_size, real_data=samples):
  rand_indices = torch.randint(0, n_samples, (batch_size, ))
  return samples[rand_indices, :]

# Training loop
device = 'cpu'
latent_dim = 2
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)
def train_gan(fake_data, num_epochs=20000, batch_size=256, latent_dim=2, device='cpu'):
    optim_G = optim.Adam(G.parameters(), lr=1e-3)
    optim_D = optim.Adam(D.parameters(), lr=1e-3)

    loss_fn = nn.BCELoss()
    n = fake_data.size()[0]
    samps = sample_unit_ball(latent_dim, num_samples=n)

    for epoch in range(1, num_epochs + 1):
        # === Train Discriminator ===
        real_data = sample_real_data(batch_size, real_data=samps).to(device)
        fake_data = sample_real_data(batch_size, real_data=fake_data).to(device)
        #fake_data = G(torch.randn(batch_size, latent_dim).to(device)).detach()

        D_real = D(real_data)
        D_fake = D(fake_data)

        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        D_loss = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)

        optim_D.zero_grad()
        D_loss.backward()
        optim_D.step()

        # === Train Generator ===
        z = sample_real_data(batch_size, real_data=fake_data).to(device)
        generated_data = G(z)
        D_generated = D(generated_data)

        G_loss = loss_fn(D_generated, torch.ones_like(D_generated))

        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()

        # Logging & visualization
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch} | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")
            with torch.no_grad():
                samples = G(Y).cpu()
                nx = samples.size()[0]
                plt.figure(figsize=(5, 5))
                plt.scatter(samples[:, 0], samples[:, 1], s=5, label='Generated')
                real = sample_unit_ball(latent_dim, num_samples=nx)
                plt.scatter(real[:, 0], real[:, 1], s=5, alpha=0.5, label='Real')
                plt.legend()
                plt.title(f"GAN Samples at Epoch {epoch}")
                plt.axis("equal")
                plt.grid(True)
                plt.show()

train_gan(Y, num_epochs=10000)

import torch

# === MMD with the Gaussian Kernel ===

def compute_pairwise_sq_dists(X, Y):
    XX = torch.mm(X, X.t())
    YY = torch.mm(Y, Y.t())
    XY = torch.mm(X, Y.t())

    X_sqnorms = XX.diag().unsqueeze(1)
    Y_sqnorms = YY.diag().unsqueeze(1)

    dists_xx = X_sqnorms + X_sqnorms.t() - 2 * XX
    dists_yy = Y_sqnorms + Y_sqnorms.t() - 2 * YY
    dists_xy = X_sqnorms + Y_sqnorms.t() - 2 * XY

    return dists_xx, dists_yy, dists_xy

def rbf_kernel(dists, bandwidth):
    return torch.exp(-dists / (2 * bandwidth ** 2))

def compute_mmd_rbf(X, Y, bandwidth=1.0):
    """Computes MMD^2 between X and Y using the RBF kernel."""
    dists_xx, dists_yy, dists_xy = compute_pairwise_sq_dists(X, Y)

    K_xx = rbf_kernel(dists_xx, bandwidth)
    K_yy = rbf_kernel(dists_yy, bandwidth)
    K_xy = rbf_kernel(dists_xy, bandwidth)

    m = X.size(0)
    n = Y.size(0)

    # unbiased estimate
    mmd = (K_xx.sum() - torch.diagonal(K_xx).sum()) / (m * (m - 1)) \
        + (K_yy.sum() - torch.diagonal(K_yy).sum()) / (n * (n - 1)) \
        - 2 * K_xy.mean()
    return mmd

# === EXAMPLE ===
torch.manual_seed(0)

# Sample from N(0, I)
X = torch.randn(100, 2)

# Sample from N(2, I)
Y = torch.randn(100, 2) + 2.0

# Compute MMD
mmd_value = compute_mmd_rbf(X, Y, bandwidth=1.0)
print(f"MMD^2 (RBF kernel, bandwidth=1.0): {mmd_value.item():.4f}")
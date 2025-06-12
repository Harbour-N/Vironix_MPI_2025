% Define the grid
x = linspace(0, 5);
y = linspace(0, 5);
[X, Y] = meshgrid(x, y);

% Define multiple point charges: [x, y, amplitude]
charges = [
    2, 2, 3;
    2, 1, 2;
    1, 1, 4;
    4, 4, 6;
];

% Parameters
sigma = 1;

% Reshape grid to vector form
Xv = X(:);
Yv = Y(:);

% Extract charge parameters
x0 = charges(:,1);
y0 = charges(:,2);
A  = charges(:,3);
n  = numel(Xv);              % Number of grid points
m  = size(charges, 1);       % Number of charges

% Compute distance squared from each charge center to all grid points
% Result: n x m matrix of squared distances
DX = Xv - reshape(x0, 1, []);
DY = Yv - reshape(y0, 1, []);
R2 = DX.^2 + DY.^2;

% Compute Gaussians for all charges at all points
Gaussians = exp(-R2 / (2 * sigma^2)) .* reshape(A, 1, []);

% Sum Gaussians over all charges, reshape back to grid
G_total = reshape(sum(Gaussians, 2), size(X));

% Plotting
figure(1);
contourf(X, Y, G_total, 50, 'LineColor', 'none');
colormap(parula);
colorbar;
hold on;
scatter(x0, y0, 100, 'kx', 'LineWidth', 2);
title('Vectorized 2D Gaussian Approximation of Multiple Point Charges');
xlabel('x');
ylabel('y');
axis equal;
legend('Gaussian Field', 'Point Charges');

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

# Set random seed for reproducibility
np.random.seed(1234)

# Number of points to generate
N = 80
M = round(N / 4)

# Generate artificial data
def generate_data():
    # First cluster (class 1)
    x1s = np.random.rand(M) * 4.5
    x2s = np.random.rand(M) * 4.5
    xt1s = np.array([[x1 + 0.5, x2 + 0.5] for x1, x2 in zip(x1s, x2s)])

    x1s = np.random.rand(M) * 4.5
    x2s = np.random.rand(M) * 4.5
    xt1s = np.vstack([xt1s, np.array([[x1 - 5, x2 - 5] for x1, x2 in zip(x1s, x2s)])])

    # Second cluster (class 0)
    x1s = np.random.rand(M) * 4.5
    x2s = np.random.rand(M) * 4.5
    xt0s = np.array([[x1 + 0.5, x2 - 5] for x1, x2 in zip(x1s, x2s)])

    x1s = np.random.rand(M) * 4.5
    x2s = np.random.rand(M) * 4.5
    xt0s = np.vstack([xt0s, np.array([[x1 - 5, x2 + 0.5] for x1, x2 in zip(x1s, x2s)])])

    # Combine all data
    xs = np.vstack([xt1s, xt0s])
    ts = np.hstack([np.ones(2 * M), np.zeros(2 * M)])

    return xt1s, xt0s, xs, ts

# Generate the data
xt1s, xt0s, xs, ts = generate_data()

# Plot data points
def plot_data():
    plt.figure(figsize=(8, 6))
    plt.scatter(xt1s[:, 0], xt1s[:, 1], c='red', label='Class 1')
    plt.scatter(xt0s[:, 0], xt0s[:, 1], c='blue', label='Class 0')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display the plot
plot_data()
#!/usr/bin/env python3
import io
import numpy as np
import pickle
from pathlib import Path
import json
from typing import Dict, Any, Tuple, List, Type, Union, Optional
import matplotlib.pyplot as plt
import matplotlib

def multivariate_normal_pdf(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """
    Compute the multivariate normal probability density function (PDF).
    """
    n_features = mean.shape[0]  # Dimension of the feature space

    # Compute the normalization constant
    det_cov = np.linalg.det(cov)  # Determinant of the covariance matrix
    norm_const = 1 / np.sqrt((2 * np.pi) ** n_features * det_cov)

    # Compute the exponent term
    diff = x - mean  # Difference between the point and the mean
    inv_cov = np.linalg.inv(cov)  # Inverse of the covariance matrix
    exponent = -0.5 * diff.T @ inv_cov @ diff

    # Return the PDF value
    return norm_const * np.exp(exponent)


def plot_conditional_risk(X, y, p_y, mu_y, V, n_classes, loss_matrix):
    x_min, x_max = np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1
    y_min, y_max = np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1
    x, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    xy = np.column_stack([x.flat, y_grid.flat])
    
    # Compute conditional risk for all points
    risks = compute_conditional_risk(xy, p_y, mu_y, V, n_classes, loss_matrix).reshape(x.shape)
    
    plt.figure()
    plt.contourf(x, y_grid, risks, levels=5, cmap='viridis')
    plt.colorbar(label="Conditional Risk")
    plt.scatter(X[:, 0], X[:, 1], c='white', edgecolor='k', alpha=0.7, label="Data Points")
    plt.title("Conditional Risk Across Observation Space")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def plot_gaussians(X, y, mu_y, V, n_classes):
    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["tab10"]  # Use Matplotlib's tab10 colormap
    colors = [matplotlib.colors.to_hex(cmap(i % 10)) for i in range(n_classes)]
    
    # Plot data points
    for c in range(n_classes):
        X_c = X[y == c]
        ax.scatter(X_c[:, 0], X_c[:, 1], alpha=0.3, label=f'Class {c}', color=colors[c])

    # Plot Gaussian contours
    x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    xy = np.column_stack([x.flat, y.flat])
    
    for c in range(n_classes):
        z = np.array([multivariate_normal_pdf(_, mean=mu_y[c], cov=V) for _ in xy]).reshape(x.shape)
        ax.contour(x, y, z, levels=5, alpha=0.7, colors=colors[c])
    
    ax.set_title("Data Points and Gaussian Distributions")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.legend()
    plt.show()
    

def save_data_to_buffer(data: Dict[Any, Any], output_file_path: str) -> None:
    """
    Save a dictionary to a binary buffer and write it to a file.
    """
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)

    output_file_path = Path(output_file_path)
    output_file_path.write_bytes(buffer.getvalue())


def load_data_from_buffer(input_file_path: str) -> Dict[Any, Any]:
    """
    Load a dictionary from a binary buffer stored in a file.
    """
    input_file_path = Path(input_file_path)
    buffer = io.BytesIO(input_file_path.read_bytes())
    buffer.seek(0)

    data = pickle.load(buffer)
    return data


def read_input(input_path: str) -> dict:
    with open(input_path, "r") as file:
        data = json.load(file)

    def convert_to_numpy(d):
        if isinstance(d, dict):
            return {k: convert_to_numpy(v) for k, v in d.items()}
        elif isinstance(d, list):
            try:
                return np.array(d)
            except ValueError:
                return d
        else:
            return d

    return convert_to_numpy(data)

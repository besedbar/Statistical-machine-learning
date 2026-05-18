#!/usr/bin/env python3
import io
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Callable

def save_result_to_buffer(alphas: np.ndarray, bias: float, output_file_path: str) -> None:
    """
    Saves the learned alphas and bias to a binary buffer file.
    
    Args:
        alphas (np.ndarray): The learned dual coefficients.
        bias (float): The learned bias term.
        output_file_path (str): Path to the output file.
    """
    buffer = io.BytesIO()
    np.savez(buffer, alphas=alphas, bias=np.array([bias]))
    buffer.seek(0)
    
    Path(output_file_path).write_bytes(buffer.getvalue())
    
def read_input(input_file_path: str) -> Dict[str, Any]:
    """
    Reads the input data for the kernel Perceptron task from a JSON file.
    
    Args:
        input_file_path (str): Path to the input JSON file.
    
    Returns:
        Dict[str, Any]: A dictionary containing the training data (X, y), 
                        kernel information, and training parameters.
    """
    with open(input_file_path, "r") as file:
        data = json.load(file)
        return {
            "X": np.array(data["X"]), 
            "y": np.array(data["y"]), 
            "kernel_type": data["kernel_type"], 
            "kernel_params": data["kernel_params"], 
            # --- THIS IS THE FIX ---
            # The input file now contains 'max_updates' instead of 'n_iter'.
            "max_updates": data["max_updates"]
        }

def visualize_decision_boundary(X: np.ndarray, y: np.ndarray, alphas: np.ndarray, bias: float, kernel_func: Callable, title="Decision Boundary"):
    """
    Generates and displays a plot of the decision boundary for a 2D dataset.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Please install it to visualize the results: pip install matplotlib")
        return

    print("Generating plot...")
    
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = np.zeros(len(grid_points))
    
    # Pre-calculate for efficiency
    ay = alphas * y

    for i, x_grid in enumerate(grid_points):
        # Calculate kernel values between the grid point and all training points
        kernel_vals = np.array([kernel_func(x_train, x_grid) for x_train in X])
        scores[i] = np.dot(ay, kernel_vals) + bias

    Z = scores.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5, levels=[-np.inf, 0, np.inf])
    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class -1', 'Class 1'])


    # Highlight points with alpha > 0, which act like support vectors
    sv_indices = alphas > 1e-5
    if np.any(sv_indices):
        plt.scatter(X[sv_indices, 0], X[sv_indices, 1], s=100,
                    facecolors='none', edgecolors='lime', linewidth=2, label='Boundary Points (alpha > 0)')
        plt.legend()


    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
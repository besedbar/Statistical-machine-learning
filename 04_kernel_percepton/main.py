#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple, Callable
from utils import read_input, save_result_to_buffer, visualize_decision_boundary

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> float:
    """
    Computes the Radial Basis Function (RBF) kernel between two vectors.

    Args:
        x1 (np.ndarray): First input vector of shape (n_features,).
        x2 (np.ndarray): Second input vector of shape (n_features,).
        gamma (float): Kernel coefficient that controls the width of the Gaussian.
            Higher values make the kernel more sensitive to small differences.

    Returns:
        float: The computed RBF kernel value between `x1` and `x2`.
    """
    # TODO: Implement the kernel
    return np.exp(-gamma*np.sum((x1 - x2) ** 2))

def polynomial_kernel(x1: np.ndarray, x2: np.ndarray, c: float, d: int) -> float:
    """
    Computes the Polynomial kernel between two vectors.

    Args:
        x1 (np.ndarray): First input vector of shape (n_features,).
        x2 (np.ndarray): Second input vector of shape (n_features,).
        c (float): Coefficient term controlling the offset of the polynomial.
        d (int): Degree of the polynomial.

    Returns:
        float: The computed Polynomial kernel value between `x1` and `x2`.
    """
    # TODO: Implement the kernel
    return (np.dot(x1, x2) + c) ** d

def kernel_perceptron(X: np.ndarray, 
                      y: np.ndarray, 
                      max_updates: int, 
                      kernel_func: Callable) -> Tuple[np.ndarray, float]:
    """
    Trains a kernel Perceptron model.
    
    Args:
        X (np.ndarray): Training data of shape (n_samples, n_features).
        y (np.ndarray): Training labels of shape (n_samples,). Labels must be -1 or 1.
        kernel_func (Callable): A kernel function that takes two vectors and returns a float.
        max_updates (int): The maximum theoretical number of updates (mistakes). You can use this value as a sanity check to make sure your code is correct.
    
    Returns:
        Tuple[np.ndarray, float]: A tuple containing the final alphas and bias.
    """
    n_samples, _ = X.shape
    alphas = np.zeros(n_samples)
    bias = 0.0

    #předpočítám si kernel:
    K = np.empty((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])

    
    #TODO: Implement training of the kernelized perceptron
    for _ in range(max_updates):
        errors = 0
        for i in range(n_samples):
            # tady prijdu o jeden for cykklus
            decision = sum(alphas * y * K[:,i]) + bias

            if y[i] * decision <= 0:
                alphas[i] += 1
                bias += y[i]
                errors += 1
        if errors == 0:
            break
                    
    return alphas, bias

def main():
    parser = argparse.ArgumentParser(description="Kernel Perceptron for automatic evaluation.")
    parser.add_argument('input_file_path', type=str, help='Path to the input JSON file')
    parser.add_argument('output_file_path', nargs='?', default=None, type=str, help='Path for the output file')
    parser.add_argument('--brute', action='store_true', help='Flag for evaluation in Brute')
    parser.add_argument('--visualize', action='store_true', help='Show a plot of the decision boundary')
    args = parser.parse_args()

    data = read_input(args.input_file_path)
    X, y, kernel_type, kernel_params, max_updates = data['X'], data['y'], data['kernel_type'], data['kernel_params'], data['max_updates']
    
    np.random.seed(42)

    if kernel_type == 'rbf':
        kernel = lambda x1, x2: rbf_kernel(x1, x2, **kernel_params)
    elif kernel_type == 'polynomial':
        kernel = lambda x1, x2: polynomial_kernel(x1, x2, **kernel_params)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    alphas, bias = kernel_perceptron(X, y, max_updates, kernel)

    if args.brute:
        save_result_to_buffer(alphas, bias, args.output_file_path)
    else:
        # Local testing logic
        print("Evaluating if the learned classifier correctly separates all training data...")
        
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kernel(X[i], X[j])
        
        scores = K @ (alphas * y) + bias
        pred_y = np.sign(scores)
        accuracy = np.mean(pred_y == y)
        print(f"Training Accuracy: {accuracy * 100:.2f}%")

        if accuracy == 1.0:
            print("✅ Test OK.")
        else:
            print("❌ Test FAILED: The model did not correctly classify all training points.")
    
    if args.visualize:
        if X.shape[1] != 2:
            print("\nVisualization is only available for 2D data.")
        else:
            title = f"Kernel Perceptron ({kernel_type} kernel)\nparams: {kernel_params}"
            visualize_decision_boundary(X, y, alphas, bias, kernel, title=title)

if __name__ == "__main__":
    main()
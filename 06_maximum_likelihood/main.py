#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import read_input, save_data_to_buffer, load_data_from_buffer, plot_gaussians, multivariate_normal_pdf
from typing import Tuple

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("input_file_path", type=str, help="Path to the input file")
parser.add_argument('--plot', action='store_true', help='Plot the data')
# These arguments will be set appropriately by Brute, even if you change them.
parser.add_argument('output_file_path', nargs='?', default=None, type=str, help='Path to the output file')
parser.add_argument("--brute", action="store_true", help="Evaluation in Brute")


def MLE_parameters(
    X: np.ndarray, y: np.ndarray, n_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate parameters for a Gaussian mixture model using Maximum Likelihood Estimation (MLE).

    Parameters
    ----------
    X : np.ndarray
        A (n_samples, n_features) array containing the feature data.
    y : np.ndarray
        A (n_samples,) array containing class labels for each data point.
    n_classes : int
        The number of distinct classes in the dataset.

    Returns
    -------
    p_y : np.ndarray
        A (n_classes,) array containing the prior probabilities for each class.
    mu_y : np.ndarray
        A (n_classes, n_features) array where each row corresponds to the mean vector of a class.
    C : np.ndarray
        A (n_features, n_features) covariance matrix shared across all classes.
    """
    n_samples, n_features = X.shape

    # Initialize arrays for priors, means, and covariance matrix
    p_y = np.zeros(n_classes)  # Priors for each class
    mu_y = np.zeros((n_classes, n_features))  # Mean vectors for each class
    C = np.zeros((n_features, n_features))  # Common covariance matrix

    # TODO: Estimate prior probabilities and class means
    for c in range(n_classes):
        X_c = X[y == c]  # Subset of data points belonging to class c
        # TODO: 1) Compute the MLE estimate of the prior for the class c
        Nc= X_c.shape[0]
        p_y[c] = Nc/n_samples
        # TODO: 2) Compute the MLE estimate of the centroid for the class c
        mu_y[c] = np.mean(X_c, axis=0)

    # Compute the MLE estimate of the covariance matrix shared by all classes
    for c in range(n_classes):
        X_c = X[y == c]  # Subset of data points belonging to class c
        # TODO: 3) Add contribution of the class c 
        diff = X_c - mu_y[c]
        C += diff.T @ diff
    C /= n_samples

    return p_y, mu_y, C


def bayes_classifier(
    x: np.ndarray,
    p_y: np.ndarray,
    mu_y: np.ndarray,
    C: np.ndarray,
    n_classes: int,
    loss_matrix: np.ndarray,
) -> Tuple[int, float]:
    """
    Perform classification using a plug-in Bayes classifier.

    Parameters
    ----------
    x : np.ndarray
        A (n_features,) array representing the data point to classify.
    p_y : np.ndarray
        A (n_classes,) array containing the prior probabilities for each class.
    mu_y : np.ndarray
        A (n_classes, n_features) array where each row corresponds to the mean vector of a class.
    C : np.ndarray
        A (n_features, n_features) covariance matrix shared across all classes.
    n_classes : int
        The number of distinct classes.
    loss_matrix : np.ndarray
        A (n_classes, n_classes) matrix where entry (i, j) represents the loss incurred
        for predicting class j when the true class is i.

    Returns
    -------
    predicted_class : int
        The class label predicted for the input data point.
    risk : float
        A float representing the conditional risks of the predicted class.
    """
    # Initialize p(x,y) for all classes
    prob_x_y = np.zeros(n_classes)

    # Compute p(x,y) for each class
    for c in range(n_classes):
        prob_x_y[c] = multivariate_normal_pdf(x, mean=mu_y[c], cov=C) * p_y[c]

    # TODO: Compute posterior probabilities p(y|x)
    # TODO: 1) Compute p(x) as: Sum over possible classes [ p(x,y) ]
    p_x = np.sum(prob_x_y)

    # TODO: 2) Use Bayes' rule to compute the posterior probabilities p(y|x)
    prob_y_given_x = prob_x_y / p_x


    # TODO: 3) Compute conditional risks for each class
    risks = np.zeros(n_classes)
    #for c_pred in range(n_classes):
    #    for c_true in range(n_classes):
    #        risks[c_pred] += loss_matrix[c_true, c_pred] * prob_y_given_x[c_true]
    risks = prob_y_given_x @ loss_matrix
    # TODO: 4) Predict the class with the minimum risk
    predicted_class = int(np.argmin(risks))
    risk = float(risks[predicted_class]) 

    return predicted_class, risk


def test_MLE(args):
    # Read the input as a dictionary
    data = read_input(args.input_file_path)
    # Load the data
    X = data["X"]
    y = data["y"]
    n_classes = data["n_classes"]
    # Estimate the priors, means and covariance matrix with Maximum Likelihood Estimation
    p_y, mu_y, C = MLE_parameters(X, y, n_classes)
    # Prepare output for comparison / saving
    student_output = {
        "class_priors": p_y,
        "class_centroids": mu_y,
        "shared_covariance_matrix": C,
    }

    # Evaluate on public instances
    if not args.brute:
        print("\nMaximum Likelihood Estimate:")
        reference_output = load_data_from_buffer(
            args.input_file_path.replace("instances", "solutions").replace(".json", "")
        )["MLE"]

        for name in reference_output.keys():
            are_identical = np.allclose(
                student_output[name], reference_output[name], rtol=1e-05, atol=1e-05
            )

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")

    return student_output


def test_bayes_classifier(args):
    # Read the input as a dictionary
    data = read_input(args.input_file_path)
    # Load the data
    X = data["X"]
    loss_matrix = data["loss_matrix"]
    n_classes = data["n_classes"]
    p_y = data["priors"]
    mu_y = data["means"]
    C = data["cov"]
    # Get predicted class and conditional risks for every observation
    predictions = []
    risks = []
    for x in X:
        pred, r = bayes_classifier(x, p_y, mu_y, C, n_classes, loss_matrix)
        predictions.append(pred)
        risks.append(r)
    predictions = np.stack(predictions)
    risks = np.stack(risks)
    # Prepare output for comparison / saving
    student_output = {"predictions": predictions, "risks": risks}

    # Evaluate on public instances
    if not args.brute:
        print("\nBayes Classifier:")
        reference_output = load_data_from_buffer(
            args.input_file_path.replace("instances", "solutions").replace(".json", "")
        )["Bayes"]

        for name in reference_output.keys():
            are_identical = np.allclose(
                student_output[name], reference_output[name], rtol=1e-05, atol=1e-05
            )

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")

    return student_output


def main(args):
    if args.plot:
        # Read the data as a dictionary
        data = read_input(args.input_file_path)
        # Load the data
        X = data["X"]        
        y = data["y"]
        n_classes = data["n_classes"]
        p_y = data["priors"]
        mu_y = data["means"]
        C = data["cov"]
        # If features are 2 dimensional, visualize the gaussians        
        if X.shape[1] == 2:
            print("Visualizing Gaussian Mixture ...")
            plot_gaussians(X, y, mu_y, C, n_classes)

    # Run student code
    MLE_output = test_MLE(args)
    Bayes_output = test_bayes_classifier(args)
    
    if args.plot:
        # Read the data as a dictionary
        data = read_input(args.input_file_path)
        # Load the data
        X = data["X"]
        loss_matrix = data["loss_matrix"]
        n_classes = data["n_classes"]
        p_y = data["priors"]
        mu_y = data["means"]
        C = data["cov"]
        
        # If features are 2 dimensional, visualize the conditional risk        
        if X.shape[1] == 2:        
            print("Visualizing Conditional Risk ...")
            x_min, x_max = np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1
            y_min, y_max = np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1
            x, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            xy = np.column_stack([x.flat, y_grid.flat])
            
            # Get risk of prediction for every observation
            risks = []
            for xy_ in xy:
                pred, r = bayes_classifier(xy_, p_y, mu_y, C, n_classes, loss_matrix)
                risks.append(r)
                
            risks = np.array(risks).reshape(x.shape)
            
            plt.figure()
            plt.contourf(x, y_grid, risks, levels=5, cmap='viridis')
            plt.colorbar(label="Conditional Risk")
            plt.scatter(X[:, 0], X[:, 1], c='white', edgecolor='k', alpha=0.7, label="Data Points")
            plt.title("Conditional Risk Across The Observation Space")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.show()
    
    # Save student output for BRUTE
    if args.brute:
        save_data_to_buffer(
            {"MLE": MLE_output, "Bayes": Bayes_output}, args.output_file_path
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

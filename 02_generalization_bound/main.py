#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple
from utils import read_input, save_list_to_path

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument('input_file_path', type=str, help='Path to the input file')
parser.add_argument('output_file_path', nargs='?', default=None, type=str, help='Path to the output file')
# These arguments will be set appropriately by Brute, even if you change them.
parser.add_argument('--brute', action='store_true', help='Evaluation in Brute')

def generalization_bound(true_y: np.ndarray, pred_y: np.ndarray, H: int, loss_matrix: np.ndarray, delta: float) -> float:
    """
    Compute the confidence interval from training data.
    
    Args:
        true_y (np.ndarray): Integer array representing the true class labels; label is an integer from 0 to Y-1.
        pred_y (np.ndarray): Integer array representing the predicted class labels; label is an integer from 0 to Y-1.   
        H (int): Number of unique hypotheses.
        loss_matrix (np.ndarray): Loss represented as a (Y,Y) shape matrix. The value loss_matrix[y,yy] represents the loss incurred when the true label is y and prediction is yy.
        delta (float): Scalar from (0,1) representing the probability of failure.
    
    Returns:
        float: The upper bound on the true risk.
    """
 
    # 1) Compute the training error
    m=len(true_y)
    # TODO: 1) Compute the empirical risk
    #R=
    R_train = 1/m*(sum(loss_matrix[true_y[i]][pred_y[i]] for i in range(m)))
    # 2) Compute epsilon, specifying how much the empirical risk on the training set can deviate from the true risk
    epsilon = np.sqrt((np.log(H) + np.log(2/delta)) / (2*m))
    # 3) Compute the upper bound, such that the true risk is smaller with probability atleast 1-delta 
    R_UB = R_train + epsilon
    return R_UB

def main(args):
    data = read_input(args.input_file_path) # Read the input data as a dictionary
    R_UB = generalization_bound(**data) # Compute the confidence interval
    print(f"Given that we selected the predictor from a hypothesis space of {data['H']} predictors,")
    print(f"the true risk of the predictor is smaller than [{R_UB:.2f}] with probability at least {1-data['delta']:.2f}\n")
    
    if args.brute:
        save_list_to_path([R_UB], args.output_file_path) # Save the result
    else:
        print("Comparing with reference solution")
        with open(args.input_file_path.replace('instances', 'solutions').replace('.json', ''), 'r') as f:
            for line in f:
                reference = list(map(float, line.split()))

        are_identical = np.allclose(np.array([R_UB]), reference[1], rtol=1e-05, atol=1e-05)
        if are_identical:
            print("Test OK")
        else:
            print("Test Failed")  

if __name__ == "__main__":            
    args = parser.parse_args()    
    main(args)

#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple
from utils import read_input, save_list_to_path

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument('input_file_path', type=str, help='Path to the input file')
# These arguments will be set appropriately by Brute, even if you change them.
parser.add_argument('output_file_path', nargs='?', default=None, type=str, help='Path to the output file')
parser.add_argument('--brute', action='store_true', help='Evaluation in Brute')

def confidence_interval(true_y: np.ndarray, 
                        pred_y: np.ndarray,  
                        loss_matrix: np.ndarray, 
                        delta: float) -> Tuple[float, float]:
    """
    Compute the confidence interval for the loss.
    
    Args:
        true_y (np.ndarray): Integer array representing the true class labels; label is an integer from 0 to Y-1.
        pred_y (np.ndarray): Integer array representing the predicted class labels; label is an integer from 0 to Y-1.   
        loss_matrix (np.ndarray): Loss represented as a (Y,Y) shape matrix. The value loss_matrix[y,yy] represents the loss incurred when the true label is y and prediction is yy.
        delta (float): Scalar from (0,1) representing the probability of failure.
    
    Returns:
        Tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    assert np.size(true_y) == np.size(pred_y)
    #Y = loss_matrix.shape[0] # Number of unique classes. Note that this means that the largest class label is Y-1.
    m=len(true_y)
    l_max=np.max(loss_matrix)
    l_min=np.min(loss_matrix)

    # TODO: 1) Compute the empirical risk
    R=1/m*(sum(loss_matrix[true_y[i]][pred_y[i]] for i in range(m)))
    # TODO: 2) Compute epsilon, specifying how much the empirical risk can deviate from the true risk
    epsilon=np.sqrt((np.log(2)-np.log(delta))/(2*m))*(l_max-l_min)
    # TODO: 3) Compute the confidence interval, in which the true risk lies with probability atleast 1-delta 
    R_L=R-epsilon
    R_R=R+epsilon
    return R_L, R_R

def main(args):
    data = read_input(args.input_file_path) # Read the input data as a dictionary
    R_L, R_R = confidence_interval(**data) # Compute the confidence interval
    print(f"The true risk is in the interval [{R_L:.2f}, {R_R:.2f}] with probability atleast {1-data['delta']}")
    
    if args.brute:
        save_list_to_path([R_L, R_R], args.output_file_path) # Save the result
    else:
        print("Comparing with reference solution")
        with open(args.input_file_path.replace('instances', 'solutions').replace('.json', ''), 'r') as f:
            for line in f:
                reference = list(map(float, line.split()))

        # If the lower bound is negative, clamp it to zero
        # We accept both: solutions that clamp the value, and solutions that do not
        if (reference[0] <= 0) and (R_L <= 0):
            reference[0] = 0
            R_L = 0
            
        are_identical = np.allclose(np.array([R_L, R_R]), reference, rtol=1e-05, atol=1e-05)
        if are_identical:
            print("Test OK")
        else:
            print("Test Failed")  

if __name__ == "__main__":            
    args = parser.parse_args()    
    main(args)

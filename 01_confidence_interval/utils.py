#!/usr/bin/env python3
import numpy as np
import json
from typing import Dict, Any, Tuple, List

def save_list_to_path(list: List, output_file_path: str) -> None:
    """
    Save a list of numbers to a text file.
    
    Args:
        list (List): The list to save.
        output_file_path (str): Path to the output file.
    """
    # Open the output file to write the results
    with open(output_file_path, 'w') as outfile:
        # Write the values separated by spaces
        outfile.write(" ".join([str(val) for val in list]) + "\n")
    
def read_input(input_file_path: str) -> Dict[str, Any]:
    """
    Read input data from a JSON file.
    
    Args:
        input_file_path (str): Path to the input JSON file.
    
    Returns:
        Dict[str, Any]: Parsed data including true_y, pred_y, Y, loss_matrix, and delta.
    """
    with open(input_file_path, "r") as file:
        data = json.load(file)
        true_y = np.array(data["true_y"]) 
        pred_y = np.array(data["pred_y"]) 
        loss_matrix = np.array(data["loss_matrix"])  
        delta = data["delta"]
        return {"true_y": true_y, 
                "pred_y": pred_y, 
                "loss_matrix": loss_matrix, 
                "delta": delta}

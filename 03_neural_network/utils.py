#!/usr/bin/env python3
import io
import numpy as np
import pickle
from pathlib import Path
import json
from typing import Dict, Any, Tuple

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
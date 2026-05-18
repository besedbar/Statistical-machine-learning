#!/usr/bin/env python3
import io
import numpy as np
from pathlib import Path
from typing import Dict, Any

def save_array_to_buffer(array: np.ndarray, output_file_path: str) -> None:
    """Save array to binary file (Brute format)."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_bytes(buffer.getvalue())
    
def load_array_from_buffer(input_file_path: str) -> np.ndarray:
    """Load array from binary file."""
    input_file_path = Path(input_file_path)
    buffer = io.BytesIO(input_file_path.read_bytes())
    buffer.seek(0)
    return np.load(buffer)

def read_input(input_file_path: str) -> Dict[str, Any]:
    """Read the packaged .npz input file."""
    with np.load(input_file_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

def stable_softmax(x):
    """
    Computes softmax(x) along the last axis in a numerically stable way.
    Used in the E-step to prevent overflow/underflow.
    """
    # Shift x by subtracting max to avoid exp() overflow
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator
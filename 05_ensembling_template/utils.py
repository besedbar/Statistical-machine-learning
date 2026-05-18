#!/usr/bin/env python3
import io
import numpy as np
import pickle
from pathlib import Path
import json
from typing import Dict, Any, Tuple, List, Type, Union, Optional
import matplotlib.pyplot as plt
from time import time
from data import Dataset


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


def evaluate_all(model: Any, data: Dataset) -> np.ndarray:
    """
    Makes predictions for all dataset samples.

    Parameters
    ----------
    model : Any
        Any model implementing the `evaluate(x)` method.
    data : Dataset
        The dataset instance containing the samples.

    Returns
    -------
    np.ndarray
        Predictions as a NumPy array.
    """
    return np.array([model.evaluate(x) for x in data.to_dict()])


def rmse(model: Any, data: Dataset) -> float:
    """
    Evaluates the Root Mean Squared Error (RMSE) on a dataset.

    Parameters
    ----------
    model : Any
        Any model implementing the `evaluate(x)` method and containing a `tattr` attribute.
    data : Dataset
        The dataset instance containing the samples.

    Returns
    -------
    float
        The RMSE value.
    """
    ys = evaluate_all(model, data)
    error = data[model.tattr] - ys
    return np.sqrt(np.mean(error**2))


def generate_plot(
    ds_train: Dataset,
    ds_test: Dataset,
    tattr: str,
    model_cls: Type[Any],
    iterate_over: str,
    iterate_values: List[Any],
    title: str,
    xlabel: str,
    rng: np.random.RandomState,
    iterate_labels: Optional[List[str]] = None,
    bayes_rmse: Optional[float] = None,
    **model_params: Any,
) -> None:
    """
    Generates a plot of training and testing RMSE over varying parameter values.

    Parameters
    ----------
    ds_train : Dataset
        The training dataset.
    ds_test : Dataset
        The testing dataset.
    tattr : str
        The name of the target attribute column.
    model_cls : Type[Any]
        The model class to use (e.g., RegressionTree, RandomForest, GradientBoostedTrees).
    iterate_over : str
        The name of the parameter to iterate over.
    iterate_values : List[Any]
        A list of values to iterate over.
    title : str
        The title of the plot.
    xlabel : str
        The x-axis label.
    rng : np.random.RandomState
        Random number generator.
    iterate_labels : Optional[List[str]], default=None
        Labels corresponding to `iterate_values`. If None, uses `iterate_values`.
    bayes_rmse : Optional[float], default=None
        If provided, plots the best achievable error as a reference.
    model_params : Any
        Additional model parameters.
    """
    if iterate_labels is None:
        iterate_coords = iterate_values
    else:
        assert len(iterate_labels) == len(
            iterate_values
        ), "Labels and values must have the same length."
        iterate_coords = range(len(iterate_labels))

    train_rmses, test_rmses = [], []
    for val in iterate_values:
        st = time()
        params = dict(model_params)
        params[iterate_over] = val
        model = model_cls(ds_train, tattr=tattr, rng=rng, **params)
        train_rmses.append(rmse(model, ds_train))
        test_rmses.append(rmse(model, ds_test))
        print(f"{title}: {iterate_over} = {val} finished in {time() - st:.2f}s")
    best = np.argmin(test_rmses)

    plt.figure()
    plt.plot(iterate_coords, train_rmses, ".-", label="train")
    plt.plot(iterate_coords, test_rmses, ".-", label="test")
    if bayes_rmse is not None:
        plt.axhline(y=bayes_rmse, linestyle="-.", label="$h^{*}(x)$")
    plt.plot(iterate_coords[best], test_rmses[best], "o", label="best")
    plt.xlabel(xlabel)
    if iterate_labels is not None:
        plt.xticks(iterate_coords, iterate_labels)
    plt.ylabel("RMSE")
    plt.title(f"Best test RMSE = {test_rmses[best]:.2f}")
    plt.suptitle(title)
    plt.legend()

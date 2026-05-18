from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Callable, List, Dict, Tuple, Union
from sklearn.model_selection import train_test_split


class Dataset:
    """
    A representation of a (subset of) dataset optimized for splitting during regression tree construction.
    Actual data are not copied; only indices are.
    """

    def __init__(
        self, df: Union[pd.DataFrame, Dataset], ix: Optional[np.ndarray] = None
    ) -> None:
        """
        Initializes the Dataset object.

        Parameters
        ----------
        df : Union[pd.DataFrame, Dataset]
            The dataset as a Pandas DataFrame or another Dataset instance. In the latter case, only metadata are copied.
        ix : Optional[np.ndarray], default=None
            Boolean index describing selected samples from the original dataset. If None, selects all samples.
        """
        if isinstance(df, pd.DataFrame):
            self.columns: List[str] = list(df.columns)
            self.cdict: Dict[str, int] = {c: i for i, c in enumerate(df.columns)}
            self.data: List[np.ndarray] = [df[c].values for c in self.columns]
        elif isinstance(df, Dataset):
            self.columns = df.columns
            self.cdict = df.cdict
            self.data = df.data
            assert (
                ix is not None
            ), "Index cannot be None when copying from another Dataset."
        self.ix: np.ndarray = (
            np.arange(len(self.data[0]), dtype=np.int64) if ix is None else ix
        )
        
    def __str__(self) -> str:
        """
        Returns a formatted string representation of the dataset.

        Returns
        -------
        str
            A string showing the dataset columns and a sample of rows.
        """
        # Create a DataFrame representation of the dataset
        df = pd.DataFrame({col: self[col] for col in self.columns})
        # Limit to the first few rows for display
        df_preview = df.head(10).to_string(index=False)
        # Add information about the number of rows
        return f"Dataset with {len(self)} rows and {len(self.columns)} columns:\n{df_preview}"


    def __getitem__(self, cname: str) -> np.ndarray:
        """
        Returns a dataset column.

        Parameters
        ----------
        cname : str
            The name of the column.

        Returns
        -------
        np.ndarray
            The specified column as a NumPy array.
        """
        return self.data[self.cdict[cname]][self.ix]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.ix)

    def to_dict(self) -> List[Dict[str, Union[int, float]]]:
        """
        Converts the dataset into a list of dictionaries for each data sample.

        Returns
        -------
        List[Dict[str, Union[int, float]]]
            A list where each dictionary represents a data sample with keys as column names.
        """
        return [{c: self.data[self.cdict[c]][i] for c in self.columns} for i in self.ix]

    def modify_col(self, cname: str, d: np.ndarray) -> Dataset:
        """
        Creates a copy of the dataset with a modified column.

        Parameters
        ----------
        cname : str
            The column name to be replaced.
        d : np.ndarray
            The new column data.

        Returns
        -------
        Dataset
            A new Dataset instance with the modified column.
        """
        assert len(self.ix) == len(self.data[0]), "Works only for unfiltered rows."
        new_dataset = Dataset(self, ix=self.ix)
        new_dataset.data = list(self.data)
        new_dataset.data[self.cdict[cname]] = d
        return new_dataset

    def filter_rows(
        self, cname: str, cond: Callable[[np.ndarray], np.ndarray]
    ) -> Dataset:
        """
        Filters rows based on a condition.

        Parameters
        ----------
        cname : str
            The column name to apply the condition on.
        cond : Callable[[np.ndarray], np.ndarray]
            A function that takes a column as input and returns a boolean mask.

        Returns
        -------
        Dataset
            A new Dataset containing only the rows that satisfy the condition.
        """
        col = self[cname]
        return Dataset(self, ix=self.ix[cond(col)])

    def sample(
        self, size: int, rng: np.random.RandomState = np.random.RandomState(1234)
    ) -> Dataset:
        """
        Generates a bootstrap sample from the dataset.

        Parameters
        ----------
        size : int
            The number of samples to include in the bootstrap sample.
        rng : np.random.RandomState, default=np.random.RandomState(1234)
            A random number generator for selecting samples.

        Returns
        -------
        Dataset
            A new Dataset instance containing the bootstrap sample.
        """
        ix = rng.choice(self.ix, size, replace=True)
        return Dataset(self, ix=ix)


def generate_sin_data(
    n: int, random_x: bool = False, scale: float = 0.0
) -> Tuple[Dataset, float]:
    """
    Generates a synthetic sine wave dataset.

    Parameters
    ----------
    n : int
        The number of data points to generate.
    random_x : bool, default=False
        If True, generates random x-values. Otherwise, uses a linspace.
    scale : float, default=0.0
        The standard deviation of noise added to the sine wave.

    Returns
    -------
    Tuple[Dataset, float]
        A tuple containing the generated Dataset and the root mean squared error (RMSE) of the noise.
    """
    rng = np.random.RandomState(1234)
    if random_x:
        X = rng.uniform(0, 2 * np.pi, n)
    else:
        X = np.linspace(0, 2 * np.pi, n)
    T = np.sin(X) + rng.normal(0, scale, size=X.shape)
    df = pd.DataFrame({"x": X, "t": T}, columns=["x", "t"])
    return Dataset(df), np.sqrt(np.mean((T - np.sin(X)) ** 2))


def generate_boston_housing(path_to_csv="housing.csv") -> Tuple[Dataset, Dataset]:
    """
    Loads the Boston housing dataset and splits it into training and test sets.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the training and test Dataset objects.
    """
    # Load the dataset from a CSV file
    df = pd.read_csv(path_to_csv)
    
    if "ID" in df.columns:
        df.drop(["ID"], axis=1, inplace=True)  # Remove unwanted column
    
    data_housing_train, data_housing_test = train_test_split(
        df, test_size=0.3, random_state=1
    )
    
    return Dataset(data_housing_train), Dataset(data_housing_test)

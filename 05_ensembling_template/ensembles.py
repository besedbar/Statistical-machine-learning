from typing import Optional, Callable, List, Dict, Union, Tuple
import numpy as np
from utils import evaluate_all
from data import Dataset
from tree import RegressionTree


class RandomForest:
    """
    A Random Forest model for regression.
    """
    def __init__(
        self,
        data: Dataset,
        tattr: str,
        xattrs: Optional[List[str]] = None,
        n_trees: int = 10,
        max_depth: Union[int, float] = np.inf,
        max_features: Callable[[int], int] = lambda n: n,
        rng: np.random.RandomState = np.random.RandomState(1),
    ) -> None:
        """
        Initializes the Random Forest model.

        Parameters
        ----------
        data : Dataset
            The training dataset.
        tattr : str
            The name of the target attribute column.
        xattrs : Optional[List[str]], default=None
            List of input attribute column names. Defaults to all columns except `tattr`.
        n_trees : int, default=10
            The number of trees in the forest.
        max_depth : Union[int, float], default=np.inf
            The maximum depth of each tree.
        max_features : Callable[[int], int], default=lambda n: n
            A function to determine the number of features considered when splitting a node.
        rng : np.random.RandomState, default=np.random.RandomState(1)
            Random number generator for sampling and splits.
        """
        self.xattrs = (
            [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        )
        self.tattr = tattr
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = rng
        self.forest = self.build_forest(data)

    def build_forest(self, data: Dataset) -> List[RegressionTree]:
        """
        Builds a forest of regression trees using bootstrap samples of the dataset.

        Parameters
        ----------
        data : Dataset
            The training dataset.

        Returns
        -------
        List[RegressionTree]
            A list of trained regression trees.
        """
        # TODO: 1) Prepare a list to hold the trained trees
        trees=[]
        
        # Train "self.n_trees" models on different bootstrapped datasets
        for _ in range(self.n_trees):
            # Generate a bootstrap dataset by calling 'data.sample(len(data), rng=self.rng)'
            bootstrap = data.sample(len(data), rng=self.rng)
            # TODO: 2) Train a tree on the bootstraped dataset
            tree = RegressionTree(
                data=bootstrap,
                tattr=self.tattr,
                xattrs=self.xattrs,
                max_depth=self.max_depth,
                max_features=self.max_features,
                rng=self.rng,
            )
            trees.append(tree)

            # You can train a regression tree by instantiating the class RegressionTree(...) with appropriate parameters
            
        # TODO: 3) Return a list of the trained trees
        ...
        return trees

    def evaluate(self, x: Dict[str, Union[int, float]]) -> float:
        """
        Predicts the target value for a single sample by averaging predictions from all trees.

        Parameters
        ----------
        x : Dict[str, Union[int, float]]
            A dictionary mapping attribute names to their values.

        Returns
        -------
        float
            The averaged prediction from the forest.
        """
        # TODO: 1) Evaluate every tree stored in self.forest
        predictions = [tree.evaluate(x) for tree in self.forest]

        ...
        # TODO: 2) Compute the mean prediction of the trees
        mean_prediction = np.mean(predictions)
        ...
        # TODO: 3) Return the prediction of the forest
        return mean_prediction
        ...
    
    
class GradientBoostedTrees:
    """
    A Gradient Boosted Trees model for regression.
    """

    def __init__(
        self,
        data: Dataset,
        tattr: str,
        xattrs: Optional[List[str]] = None,
        n_trees: int = 10,
        max_depth: int = 1,
        beta: float = 0.1,
        rng: np.random.RandomState = np.random.RandomState(1),
    ) -> None:
        """
        Initializes the Gradient Boosted Trees model.

        Parameters
        ----------
        data : Dataset
            The training dataset.
        tattr : str
            The name of the target attribute column.
        xattrs : Optional[List[str]], default=None
            List of input attribute column names. Defaults to all columns except `tattr`.
        n_trees : int, default=10
            The number of trees to construct.
        max_depth : int, default=1
            Maximum depth of each tree.
        beta : float, default=0.1
            The learning rate.
        rng : np.random.RandomState, default=np.random.RandomState(1)
            Random number generator for sampling and splits.
        """
        self.xattrs = (
            [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        )
        self.tattr = tattr
        self.max_depth = max_depth
        self.beta = beta
        self.n_trees = n_trees
        self.rng = rng
        self.trees, self.betas = self.build_gbm(data)

    def build_gbm(self, data: Dataset) -> Tuple[List[RegressionTree], List[float]]:
        """
        Builds the gradient boosting model.

        Parameters
        ----------
        data : Dataset
            The dataset for training the model.

        Returns
        -------
        Tuple[List[RegressionTree], List[float]]
            A tuple containing the list of trees and their corresponding learning rates.
        """
        # List to store the weak learners
        # We initialize the GBM with a tree of depth 0; this amounts to having a constant prediction as the initialization
        trees = [            
            RegressionTree(data, xattrs=self.xattrs, tattr=self.tattr, max_depth=0)
        ]
        # Compute the initial predictions f(x) for all the data points
        # We will iteratively update the predictions as more weak learners are added to the ensemble
        ys = evaluate_all(trees[0], data)
        # List to store the learning rates for each tree in the ensemble
        betas = [1.0]
        # Target values from the dataset
        ts = data[self.tattr]

        # TODO: Build "self.n_trees" additional trees iteratively to correct residual errors

        for k in range(self.n_trees):
            # TODO: 1) Compute the residuals (negative gradients) for the squared loss
            residuals = ts - ys
            
            # Create a copy of the Dataset, replacing the values in the target column with the residuals
            # We will train subsequent models to "fit" the residuals
            residual_data = data.modify_col(self.tattr, residuals)
            
            # Train a weak learner to fit the residuals (to correct the errors of the current model)
            # You can train a regression tree as the weak learner by instantiating the class RegressionTree(...) 
            tree = RegressionTree(
                residual_data,
                xattrs=self.xattrs,
                tattr=self.tattr,
                max_depth=self.max_depth,
                rng=self.rng,
            )
            # TODO: 2) Compute the predictions of the new weak learner for all samples
            deltas = evaluate_all(tree, data)
            
            # TODO: 3) Update the overall predictions by incorporating the corrections from the new weak learner
            # I.e. set f(x) <- f(x) + beta * tree(x) for every sample
            ys += self.beta * deltas
            
            # Add the new weak learner and its learning rate to the ensemble
            trees.append(tree)
            betas.append(self.beta)
            
        return trees, betas

    def evaluate(self, x: Dict[str, Union[int, float]]) -> float:
        """
        Predicts the target value for a single sample.

        Parameters
        ----------
        x : Dict[str, Union[int, float]]
            A dictionary mapping attribute names to their values.

        Returns
        -------
        float
            The predicted target value.
        """
        # TODO: 1) Evaluate every weak learner stored in self.trees
        predictions = [tree.evaluate(x) for tree in self.trees]
        # TODO: 2) Compute the weighed mean prediction of the weak learners, where the weights are the learning rates used in training 
        weighted_prediction = sum(
            beta * pred for beta, pred in zip(self.betas, predictions)
        )
        # TODO: 3) Return the prediction of the GBM
        return weighted_prediction
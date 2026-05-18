from __future__ import annotations
from typing import Union, List, Dict, Optional, Callable
from collections import deque
import numpy as np
from data import Dataset


class DecisionNode:
    """
    Represents an internal decision node in a regression tree.
    """

    def __init__(
        self,
        attr: str,
        value: Union[str, float],
        left: DecisionNode,
        right: DecisionNode,
    ) -> None:
        """
        Initializes a decision node.

        Parameters
        ----------
        attr : str
            The splitting attribute.
        value : Union[str, float]
            The splitting value for the attribute.
        left : DecisionNode
            The left child node.
        right : DecisionNode
            The right child node.
        """
        self.attr = attr
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x: Dict[str, Union[int, float]]) -> Union["DecisionNode", float]:
        """
        Evaluates the node by traversing the tree based on input attributes.

        Parameters
        ----------
        x : Dict[str, Union[int, float]]
            A dictionary mapping attribute names to their values.

        Returns
        -------
        Union[DecisionNode, float]
            The child node or prediction result.
        """
        if isinstance(self.value, str):
            return (
                self.left.evaluate(x)
                if x[self.attr] == self.value
                else self.right.evaluate(x)
            )
        return (
            self.left.evaluate(x)
            if x[self.attr] <= self.value
            else self.right.evaluate(x)
        )

    def get_nodes(self) -> List["DecisionNode"]:
        """
        Returns all nodes in the subtree rooted at this node.

        Returns
        -------
        List[DecisionNode]
            A list of all nodes in the subtree.
        """
        nodes = []
        queue = deque([self])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            if isinstance(node, DecisionNode):
                queue.append(node.left)
                queue.append(node.right)
        return nodes

    def __str__(self) -> str:
        """
        String representation of the decision node.

        Returns
        -------
        str
            A string describing the node's splitting condition.
        """
        return (
            f'{self.attr}=="{self.value}"'
            if isinstance(self.value, str)
            else f"{self.attr}<={self.value:.2f}"
        )


class LeafNode:
    """
    Represents a leaf node in a regression tree.
    """

    def __init__(self, response: float) -> None:
        """
        Initializes a leaf node.

        Parameters
        ----------
        response : float
            The prediction value for the leaf.
        """
        self.response = response

    def evaluate(self, x: Dict[str, Union[int, float]]) -> float:
        """
        Returns the prediction value of the leaf node.

        Parameters
        ----------
        x : Dict[str, Union[int, float]]
            Input attributes (ignored for leaf nodes).

        Returns
        -------
        float
            The prediction value.
        """
        return self.response

    def get_nodes(self) -> List["LeafNode"]:
        """
        Returns the leaf node itself.

        Returns
        -------
        List[LeafNode]
            A list containing the current leaf node.
        """
        return [self]

    def __str__(self) -> str:
        """
        String representation of the leaf node.

        Returns
        -------
        str
            The prediction value as a string.
        """
        return f"{self.response:.2f}"


class RegressionTree:
    """
    A regression tree model for predicting numerical targets.
    """

    def __init__(
        self,
        data: Dataset,
        tattr: str,
        xattrs: Optional[List[str]] = None,
        max_depth: int = 5,
        max_features: Callable[[int], int] = lambda n: n,
        rng: np.random.RandomState = np.random.RandomState(1),
    ) -> None:
        """
        Initializes a regression tree and builds the model.

        Parameters
        ----------
        data : Dataset
            The training dataset.
        tattr : str
            The name of the target attribute column.
        xattrs : Optional[List[str]], default=None
            A list of input attribute column names. Defaults to all columns except `tattr`.
        max_depth : int, default=5
            Maximum tree depth.
        max_features : Callable[[int], int], default=lambda n: n
            A function that determines the number of features to consider when splitting a node.
        rng : np.random.RandomState, default=np.random.RandomState(1)
            Random number generator for feature sampling.
        """
        self.xattrs = (
            [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        )
        self.tattr = tattr
        self.max_features = int(np.ceil(max_features(len(self.xattrs))))
        self.rng = rng
        self.root = self.build_tree(data, self.impurity(data), max_depth=max_depth)

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
        return self.root.evaluate(x)

    def impurity(self, data: Dataset) -> float:
        """
        Computes the impurity (squared loss) for a constant mean model.

        Parameters
        ----------
        data : Dataset
            The dataset to compute impurity for.

        Returns
        -------
        float
            The impurity value.
        """
        if len(data) == 0:
            return 0.0
        t = data[self.tattr]
        return np.sum((t - t.mean()) ** 2)

    def build_tree(
        self, data: Dataset, impurity: float, max_depth: int
    ) -> Union[DecisionNode, LeafNode]:
        """
        Recursively builds the regression tree.

        Parameters
        ----------
        data : Dataset
            The dataset to build the tree from.
        impurity : float
            The impurity of the dataset.
        max_depth : int
            The maximum depth allowed for the tree.

        Returns
        -------
        Union[DecisionNode, LeafNode]
            The root node of the tree.
        """
        if max_depth > 0:
            best_impurity = impurity
            best_xattr, best_val = None, None
            best_data_l, best_data_r = None, None
            best_impurity_l, best_impurity_r = None, None

            xattrs = self.rng.choice(self.xattrs, self.max_features, replace=False)
            for xattr in xattrs:
                vals = np.unique(data[xattr])
                if len(vals) <= 1:
                    continue
                for val in vals:
                    if isinstance(val, str):
                        data_l = data.filter_rows(xattr, lambda a: a == val)
                        data_r = data.filter_rows(xattr, lambda a: a != val)
                    else:
                        data_l = data.filter_rows(xattr, lambda a: a <= val)
                        data_r = data.filter_rows(xattr, lambda a: a > val)

                    impurity_l = self.impurity(data_l)
                    impurity_r = self.impurity(data_r)
                    split_impurity = impurity_l + impurity_r

                    if (
                        split_impurity < best_impurity
                        and len(data_l) > 0
                        and len(data_r) > 0
                    ):
                        best_impurity, best_xattr, best_val = split_impurity, xattr, val
                        best_data_l, best_data_r = data_l, data_r
                        best_impurity_l, best_impurity_r = impurity_l, impurity_r

            if best_impurity < impurity:
                return DecisionNode(
                    best_xattr,
                    best_val,
                    self.build_tree(best_data_l, best_impurity_l, max_depth - 1),
                    self.build_tree(best_data_r, best_impurity_r, max_depth - 1),
                )
        return LeafNode(data[self.tattr].mean())

import numpy as np
from typing import Optional, Union

class CART:
    """
    A class to implement a CART (Classification and Regression Tree) algorithm.

    Attributes:
        depth_limit (int): Maximum depth of the tree.
        min_crit (float): Minimum criterion value for pruning.
        criterion (str): Impurity measure ('gini' or 'entropy').
    """

    def __init__(self, depth_limit: int = 5, min_crit: float = 0.1, criterion: str = 'gini') -> None:
        """
        Initialize the CART object with hyperparameters.

        Args:
            depth_limit (int): Maximum depth of the tree.
            min_crit (float): Minimum criterion value for pruning.
            criterion (str): Impurity measure ('gini' or 'entropy').
        """
        self.root: Optional[CART] = None
        self.depth_limit: int = depth_limit
        self.min_crit: float = min_crit
        self.criterion: str = criterion
        self.feature: Optional[int] = None
        self.label: Optional[Union[float, int]] = None
        self.nsamp: Optional[int] = None
        self.gain: Optional[float] = None
        self.left: Optional['CART'] = None
        self.right: Optional['CART'] = None
        self.depth: int = 0
        self.threshold: Optional[float] = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the labels for a given set of features.

        Args:
            features (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return np.array([self.root._predict(feature) for feature in features])

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Fit the CART model to the provided features and target.

        Args:
            features (np.ndarray): Input features.
            target (np.ndarray): Target values or labels.
        """
        self.root = CART()
        self.root._build_tree(features, target)

    def _build_tree(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Recursively build the decision tree.

        Args:
            features (np.ndarray): Input features.
            target (np.ndarray): Target values or labels.
        """
        self.nsamp = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        if self.depth >= self.depth_limit or len(np.unique(target)) == 1:
            if len(np.unique(target)) == 1:
                self.label = target[0]
            else:
                if self.criterion in ["gini", "entropy"]:
                    class_counts = [(cls, (target == cls).sum()) for cls in np.unique(target)]
                    self.label = max(class_counts, key=lambda item: item[1])[0]
                else:
                    self.label = np.mean(target)
            return

        highest_gain = 0.0
        best_fit_feature, best_threshold = None, None

        if self.criterion in ["gini", "entropy"]:
            class_counts = [(cls, (target == cls).sum()) for cls in np.unique(target)]
            self.label = max(class_counts, key=lambda item: item[1])[0]
        else:
            self.label = np.mean(target)

        impurity = self._calculate_impurity(target)
        for c in range(features.shape[1]):
            unique_features = np.unique(features[:, c])
            thresholds = (unique_features[:-1] + unique_features[1:]) / 2

            for t in thresholds:
                right = target[features[:, c] > t]
                left = target[features[:, c] <= t]

                imp_r = self._calculate_impurity(right)
                imp_l = self._calculate_impurity(left)

                right_samples = right.shape[0] / self.nsamp
                left_samples = left.shape[0] / self.nsamp

                gain = impurity - (right_samples * imp_r + left_samples * imp_l)
                if gain > highest_gain:
                    highest_gain = gain
                    best_fit_feature = c
                    best_threshold = t
        if highest_gain == 0 or best_fit_feature is None or best_threshold is None:
            return
        self.feature = best_fit_feature
        self.gain = highest_gain
        self.threshold = best_threshold
        self._split(features, target)

    def _calc_gini_index(self, target: np.ndarray) -> float:
        """
        Calculate the Gini index for a given target distribution.

        Args:
            target (np.ndarray): Target values or labels.

        Returns:
            float: Gini index.
        """
        return 1.0 - sum((len(target[target == c]) / target.shape[0]) ** 2 for c in np.unique(target))

    def _calc_entropy(self, target: np.ndarray) -> float:
        """
        Calculate the entropy for a given target distribution.

        Args:
            target (np.ndarray): Target values or labels.

        Returns:
            float: Entropy.
        """
        return -sum(
            (p := len(target[target == c]) / target.shape[0]) * np.log2(p) if p > 0 else 0
            for c in np.unique(target)
        )

    def _calculate_impurity(self, target: np.ndarray) -> float:
        """
        Calculate impurity based on the criterion.

        Args:
            target (np.ndarray): Target values or labels.

        Returns:
            float: Impurity value.
        """
        if self.criterion == 'gini':
            return self._calc_gini_index(target)
        return self._calc_entropy(target)

    def _split(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Split the data into left and right branches based on the best threshold.

        Args:
            features (np.ndarray): Input features.
            target (np.ndarray): Target values or labels.
        """
        right_mask = features[:, self.feature] > self.threshold
        left_mask = ~right_mask

        right_features, right_target = features[right_mask], target[right_mask]
        left_features, left_target = features[left_mask], target[left_mask]

        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._build_tree(right_features, right_target)

        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._build_tree(left_features, left_target)

    def _predict(self, data: np.ndarray) -> Union[float, int]:
        """
        Recursively predict the label for a single data instance.

        Args:
            data (np.ndarray): A single data instance.

        Returns:
            float or int: Predicted label or value.
        """
        if self.feature is not None:
            if data[self.feature] > self.threshold:
                return self.right._predict(data)
            return self.left._predict(data)
        return self.label

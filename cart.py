"""
UMA Semestr Ziomowy 2024/2025
Adrian Zalewski
Juliusz Kuzyka
cart.py - Implementacja drzewa decyzyjnego CART
"""

import numpy as np
from typing import Optional, Union

class CART:

    def __init__(self, depth_limit: int = 5, criterion: str = 'gini') -> None:

        self.root: Optional[CART] = None
        self.depth_limit: int = depth_limit
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
        arr = []
        for feature in features:
            arr.append(self.root._predict(feature))
        return np.array(arr)

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.root = CART(depth_limit=self.depth_limit, criterion=self.criterion)
        self.root._build_tree(features, target)

    def _build_tree(self, features: np.ndarray, target: np.ndarray) -> None:
        self.nsamp = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        if self.depth >= self.depth_limit:
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
        Calculate Gini index of the target with formula: 1 - sum(p^2)
        """
        return 1.0 - sum((len(target[target == c]) / target.shape[0]) ** 2 for c in np.unique(target))

    def _calc_entropy(self, target: np.ndarray) -> float:
        """
        Calculate entropy of the target with forumla: sum(-p * log2(p))
        """
        entropy = 0.0
        for c in np.unique(target):
            p = len(target[target == c]) / target.shape[0]
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_impurity(self, target: np.ndarray) -> float:
        if self.criterion == 'gini':
            return self._calc_gini_index(target)
        return self._calc_entropy(target)

    def _split(self, features: np.ndarray, target: np.ndarray) -> None:
        right_mask = features[:, self.feature] > self.threshold
        left_mask = ~right_mask

        right_features, right_target = features[right_mask], target[right_mask]
        left_features, left_target = features[left_mask], target[left_mask]

        self.right = CART(depth_limit=self.depth_limit, criterion=self.criterion)
        self.right.depth = self.depth + 1
        self.right._build_tree(right_features, right_target)

        self.left = CART(depth_limit=self.depth_limit, criterion=self.criterion)
        self.left.depth = self.depth + 1
        self.left._build_tree(left_features, left_target)

    def _predict(self, data: np.ndarray) -> Union[float, int]:
        if self.feature is not None:
            if data[self.feature] > self.threshold:
                return self.right._predict(data)
            return self.left._predict(data)
        return self.label

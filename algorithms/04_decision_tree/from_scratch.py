import numpy as np

from ml_from_scratch.base_model import BaseModel


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


def gini(y):
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity = impurity - p**2
    return impurity


def best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_samples = X.shape[0]
    n_features = X.shape[1]
    parent_gini = gini(y)

    for feature in range(n_features):
        values = np.unique(X[:, feature])

        for threshold in values:
            left_side = X[:, feature] <= threshold
            right_side = X[:, feature] > threshold

            if len(y[left_side]) == 0 or len(y[right_side]) == 0:
                continue

            n_left = len(y[left_side])
            n_right = len(y[right_side])

            gini_left = gini(y[left_side])
            gini_right = gini(y[right_side])

            weighted_gini = (n_left/n_samples)*gini_left + (n_right/n_samples)*gini_right
            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def build_tree(X, y, depth=0, max_depth=5):
    n_classes = len(np.unique(y))

    if n_classes == 1 or depth >= max_depth or len(y) < 2:
        values, counts = np.unique(y, return_counts=True)
        leaf_value = values[np.argmax(counts)]
        return Node(value=leaf_value)

    feature, threshold, gain = best_split(X, y)

    if feature is None or gain <= 0:
        values, counts = np.unique(y, return_counts=True)
        leaf_value = values[np.argmax(counts)]
        return Node(value=leaf_value)

    left_side = X[:, feature] <= threshold
    right_side = X[:, feature] > threshold

    left_child = build_tree(X[left_side], y[left_side], depth+1, max_depth)
    right_child = build_tree(X[right_side], y[right_side], depth+1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)


class DecisionTree(BaseModel):

    def __init__(self, max_depth=5):
        super().__init__()
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        y = y.flatten()
        self.root = build_tree(X, y, 0, self.max_depth)
        self.is_fitted = True
        return self

    def predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        self._check_is_fitted()
        predictions = []
        for x in X:
            pred = self.predict_single(x, self.root)
            predictions.append(pred)
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score

    np.random.seed(42)

    n_samples = 100

    cgpa = 5 + 5 * np.random.rand(n_samples,1)
    internship = np.random.randint(0,2,(n_samples,1))
    projects = np.random.randint(0,6,(n_samples,1))

    z = (cgpa - 7.5) + (internship * 1.5) + (projects * 0.3)
    probability = 1/(1+np.exp(-z))
    random_values = np.random.rand(n_samples,1)
    placed = (probability > random_values).astype(int)

    X = np.hstack((cgpa, internship, projects))

    model = DecisionTree(max_depth=5)
    model.fit(X, placed)

    predictions = model.predict(X)
    print(predictions[:10])

    actual = placed.reshape(-1)
    accuracy = np.mean(predictions == actual)
    print("accuracy:", accuracy)

    model_shallow = DecisionTree(max_depth=2)
    model_shallow.fit(X, placed)
    predictions_shallow = model_shallow.predict(X)

    accuracy_shallow = accuracy_score(placed, predictions_shallow)
    print("Shallow Tree Accuracy:", accuracy_shallow)
"""
K-Nearest Neighbors (KNN) implemented using NumPy.
Unlike other models, this one doesn't really "train" - it just stores
the training data and does the actual work at prediction time.
"""
import numpy as np
from ml_from_scratch.base_model import BaseModel

class KNN(BaseModel):

    def __init__(self, k = 5):
        super().__init__()
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.reshape(-1)
        self.is_fitted = True
        return self

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))

    def predict_single(self, x):
        distance = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(x, self.X_train[i])
            distance.append((dist, self.y_train[i]))

        distance.sort(key=lambda pair: pair[0])

        k_nearest_labels = []
        for i in range(self.k):
            k_nearest_labels.append(distance[i][1])

        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        best_label = None
        best_count = 0
        for label, count in label_count.items():
            if count > best_count:
                best_count = count
                best_label = label

        return best_label

    def predict(self, X):
        self._check_is_fitted()
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return np.array(predictions)


if __name__ == "__main__":
    np.random.seed(42)

    cgpa = 5 + 5 * np.random.rand(100,1)
    z = cgpa - 7.5
    probability = 1/(1+np.exp(-z))
    random_values = np.random.rand(100,1)
    placed = (probability > random_values).astype(int)

    model = KNN(k=5)
    model.fit(cgpa, placed)

    predictions = model.predict(cgpa)
    print(predictions[:10])
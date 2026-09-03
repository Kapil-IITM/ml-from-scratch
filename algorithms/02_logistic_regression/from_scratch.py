"""
Logistic Regression implemented using NumPy, trained with Gradient Descent.
Used for classification problems.
"""
import numpy as np
from ml_from_scratch.base_model import BaseModel

class LogisticRegression(BaseModel):

    def __init__(self, learning_rate = 0.1, n_iterations = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros((n_features,1))
        self.bias = 0.0
        y = y.reshape(-1,1)

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weight) + self.bias
            predictions = self.sigmoid(z)

            error = predictions - y

            cost = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions)) # log loss
            self.cost_history.append(cost)

            weight_gradient = (1/n_samples) * np.dot(X.T, error)
            bias_gradient = (1/n_samples) * np.sum(error)

            self.weight -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

        self.is_fitted = True
        return self

    def predict_probability(self, X):
        self._check_is_fitted()
        z = np.dot(X, self.weight) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(42)

    cgpa = 5 + 5 * np.random.rand(100,1)
    z = cgpa - 7.5
    probability = 1/(1+np.exp(-z))
    random_values = np.random.rand(100,1)
    placed = (probability > random_values).astype(int)

    model = LogisticRegression(learning_rate=0.1, n_iterations=5000)
    model.fit(cgpa, placed)

    print("Learned weight:", model.weight[0][0])
    print("Learned bias:", model.bias)
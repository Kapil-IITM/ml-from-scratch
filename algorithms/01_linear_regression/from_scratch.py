"""
Linear Regression implemented using Numpy, trained with Gradient Descent.
Follows a scikit-learn style .fit() / .predict() interface

"""
import numpy as np 
from ml_from_scratch.base_model import BaseModel

class LinearRegression(BaseModel):
    def __init__(self, learning_rate = 0.1, n_iterations = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # cost recorded at each iteration


    def fit(self,X ,y):
        n_samples , n_features = X.shape

        self.weights = np.zeros((n_features,1))
        self.bias = 0.0

        y = y.reshape(-1,1)

        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) +  self.bias
            error = predictions - y

            cost = np.mean(error**2) # Mean Squared Error
            self.cost_history.append(cost)

            # Gradient of the cost function w.r.t weights and bias
            weight_gradient = (2/n_samples) * np.dot(X.T, error)
            bias_gradient = (2/n_samples) * np.sum(error)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

        self.is_fitted = True
        return self
    def predict(self, X):
        self._check_is_fitted()
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    np.random.seed(42)

    X_demo = 2 * np.random.rand(100,1)
    y_demo = 4 + 3*X_demo + np.random.randn(100,1)

    model = LinearRegression(learning_rate=0.1, n_iterations= 1000)
    model.fit(X_demo, y_demo)

    print("Learned weight:->", model.weights[0][0])
    print("Learned bias:->", model.bias)
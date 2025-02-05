import numpy as np
import pandas as pd


def load_data(file_path):
    """
    Load the dataset from a CSV file and split into features (X) and target (y).
    """
    # Load data
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    # Features and target
    X = data.drop(columns=["disease_score"], axis=1).values  # Drop the target column
    y = data["disease_score"].values  # Target variable

    # Add a bias term (x0 = 1) to X for the intercept
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of 1s to X
    return X, y

def compute_hypothesis(X, theta):
    """
    Compute the hypothesis (predicted values): h_theta(x) = X * theta
    """
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    """
    Compute the cost function: J(theta) = (1/2m) * sum((h_theta(x) - y)^2)
    """
    m = len(y)  # Number of training examples
    predictions = compute_hypothesis(X, theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost


def compute_gradient(X, y, theta):
    """
    Compute the gradient (partial derivatives of the cost function w.r.t. theta).
    """
    m = len(y)  # Number of training examples
    predictions = compute_hypothesis(X, theta)
    errors = predictions - y
    gradient = (1 / m) * np.dot(X.T, errors)
    return gradient


def update_parameters(theta, gradient, learning_rate):
    """
    Update the parameters (theta) using gradient descent.
    """
    return theta - learning_rate * gradient


def main():
    # Load the data
    file_path = "/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv"
    X, y = load_data(file_path)

    # Initialize parameters (theta) to zeros
    theta = np.zeros(X.shape[1])  # Number of features (including bias term)

    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000  # Number of iterations for gradient descent

    # Perform gradient descent
    print("Starting gradient descent...")
    for i in range(num_iterations):
        # Compute the gradient
        gradient = compute_gradient(X, y, theta)

        # Update the parameters
        theta = update_parameters(theta, gradient, learning_rate)

        # Compute the cost (optional: print every 100 iterations)
        if i % 100 == 0:
            cost = compute_cost(X, y, theta)
            print(f"Iteration {i}: Cost = {cost:.4f}")

    print("Gradient descent complete!")
    print("Final parameters (theta):", theta)

    # Make predictions
    predictions = compute_hypothesis(X, theta)
    print("First 5 predictions:", predictions[:5])
    print("First 5 actual values:", y[:5])


if __name__ == "__main__":
    main()

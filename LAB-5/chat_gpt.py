import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data():
    # Load the data
    df = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = df["disease_score"].values
    return X, y1


def train_testsample_division(x, y):
    data_split = int(x.shape[0] * 0.70)
    return (x[:data_split], x[data_split:], y[:data_split], y[data_split:])


def update_params(theta, alpha, gradient):
    # Update the parameters using the gradient
    return theta - alpha * gradient


def compute_cost(h, y):
    # Compute the Mean Squared Error
    return np.sum((h - y) ** 2) / (2 * len(y))


def compute_gradient(x, y, theta):
    # Compute gradient for a single data point
    h = np.dot(x, theta)
    gradient = (h - y) * x
    return gradient


def stochastic_gradient_descent(x_train, y_train, theta, alpha=0.01, epochs=100):
    m = x_train.shape[0]
    cost_history = []

    for epoch in range(epochs):
        # Shuffle data to ensure randomness
        indices = np.arange(m)
        np.random.shuffle(indices)

        for i in indices:
            # Select a single data point
            x_i = x_train[i, :].reshape(1, -1)
            y_i = y_train[i]

            # Compute gradient and update parameters
            gradient = compute_gradient(x_i, y_i, theta)
            theta = update_params(theta, alpha, gradient)

        # Compute cost for the current epoch
        h = np.dot(x_train, theta)
        cost = compute_cost(h, y_train)
        cost_history.append(cost)

        print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost}")

    return theta, cost_history


def main():
    # Load and prepare the data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_testsample_division(X, y)

    # Normalize data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Add a bias column
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Initialize parameters
    theta = np.zeros(X_train.shape[1])

    # Train the model using Stochastic Gradient Descent
    theta, cost_history = stochastic_gradient_descent(X_train, y_train, theta, alpha=0.01, epochs=100)

    # Evaluate model performance
    predictions = np.dot(X_test, theta)
    r2 = 1 - (np.sum((predictions - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"Final R^2 score: {r2}")

    # Plot cost history
    plt.plot(range(len(cost_history)), cost_history, label="Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost vs Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

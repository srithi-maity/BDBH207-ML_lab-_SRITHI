import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


def load_data():
    # Load dataset
    df = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
    y = df["disease_score"].values
    return X, y


def train_test_split(x, y, train_size=0.7):
    data = int(x.shape[0] * train_size)
    return x[:data], x[data:], y[:data], y[data:]


def hypothesis_function(theta, x):
    return np.dot(x, theta)


def cost_function(h_theta_x, y):
    return np.mean((h_theta_x - y) ** 2) / 2


def compute_errors(h_theta_x, y):
    return h_theta_x - y


def calculate_gradient(error, x):
    return error * x


def update_params(theta, gradient, alpha):
    return theta - alpha * gradient


def stochastic_gradient_descent(x_train, y_train, theta, alpha=0.001, iteration=1000000):
    m = x_train.shape[0]
    cost_history = []

    for itr in range(iteration):




        for i in range(random.randint(0,m)):

            x = x_train[i, :].reshape(1, -1)  # Shape (1, n+1)
            y = y_train[i]

            # Compute hypothesis
            h_theta_x = hypothesis_function(theta, x.flatten())  # Flatten x to match theta
            error = compute_errors(h_theta_x, y)

            # Compute gradient and update parameters
            gradient = calculate_gradient(error, x.flatten())
            theta = update_params(theta, gradient, alpha)

            # Compute cost for the current epoch
        h_theta_x_epoch = np.dot(x_train, theta)  # Use the entire training set
        cost = cost_function(h_theta_x_epoch, y_train)
        cost_history.append(cost)
        print(f"final Cost: {cost}")

    return theta, cost_history

#
# def stochastic_gradient_descent(x_train, y_train, theta, alpha=0.001, iteration=1000):
#     m = x_train.shape[0]
#     cost_history = []
#
#     for itr in range(iteration):
#             # Shuffle data
#         indices = np.arange(m)
#             # np.random.shuffle(indices)
#
#         for i in indices:
#                 # Select a single data point
#             x = x_train[i, :].reshape(1, -1)  # Shape (1, n+1)
#             y = y_train[i]
#
#                 # Compute hypothesis
#             h_theta_x = hypothesis_function(theta, x.flatten())  # Flatten x to match theta
#             error = compute_errors(h_theta_x, y)
#
#                 # Compute gradient and update parameters
#             gradient = calculate_gradient(error, x.flatten())
#             theta = update_params(theta, gradient, alpha)
#
#             # Compute cost for the current epoch
#         h_theta_x_epoch = np.dot(x_train, theta)  # Use the entire training set
#         cost = cost_function(h_theta_x_epoch, y_train)
#         cost_history.append(cost)
#         print(f" Cost: {cost}")
#
#     return theta, cost_history








def main():
    # Load and prepare the data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

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
    theta, cost_history = stochastic_gradient_descent(X_train, y_train, theta, alpha=0.01, iteration=1000)

    # Evaluate model performance
    predictions = np.dot(X_test, theta)
    r2 = 1 - (np.sum((predictions - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"Final R^2 score: {r2}")

    # Plot cost history
    plt.plot(range(len(cost_history)), cost_history, label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

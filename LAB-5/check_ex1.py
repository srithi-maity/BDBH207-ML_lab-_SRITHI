#Implement Stochastic Gradient Descent algorithm from scratch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from chat_gpt import compute_gradient
from ex1 import h_theta


def load_data():
    df = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = df["disease_score"].values
    # y2 = df["disease_score_fluct"].values
    return X, y1

def train_test_split(x,y):
    data = int(x.shape[0] * 0.70)
    # print(data)
    return (x[:data],x[data:],y[:data],y[data:])
def hypothesis_function(theta,x):
    # t=np.transpose(theta)
    print(theta.shape)
    h_theta_x= np.dot(theta,x)
    return h_theta_x

def cost_function(h_theta_x,y):
    c_f=((h_theta_x - y )**2)/2
    return c_f

def compute_errors(h_theta_x,y):
    error=h_theta_x - y
    return error

def calculate_gradient(error,x):
    gradient= error*x
    return gradient

def update_params(theta, gradient, alpha):
    # Update the parameters using the gradient
    up_para= theta - alpha * gradient
    return up_para

def stochastic_gradient_descent(x_train, y_train, theta, alpha=0.001, iteration=100):
    # for itr in range(iteration):
    m = x_train.shape[0]
    cost_history = []

    # for epoch in range(epochs):
        # Shuffle data to ensure randomness
    indices = np.arange(m)
        # np.random.shuffle(indices)

    for i in indices:
            # Select a single data point
        x = x_train[i, :].reshape(1, -1)
        y = y_train[i]

        h_theta_x=hypothesis_function(theta,x)
        c_f=cost_function(h_theta_x,y)
        error=compute_errors(h_theta_x,y)
        gradient=calculate_gradient(error,x)
        theta = update_params(theta, alpha, gradient)

        #     # Compute gradient and update parameters
        # gradient = compute_gradient(x_i, y_i, theta)
        # theta = update_params(theta, alpha, gradient)
        #
        #     # Compute cost for the current epoch
        h_theta_x = np.dot(x_train, theta)
        cost = cost_function(h_theta_x, y_train)
        cost_history.append(cost)

        print(f" Cost: {cost}")

    return theta, cost_history







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
    cost_history = stochastic_gradient_descent(X_train, y_train, theta)

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
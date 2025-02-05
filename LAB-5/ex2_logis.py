import numpy as np
import pandas as pd

def load_data():
    # Load dataset
    df = pd.read_csv("/home/ibab/Downloads/archive/data.csv")
    X = df.drop(columns=["diagnosis", "id"]).values
    df["diagnosis_binary"] = df["diagnosis"].map({'M': 1, 'B': 0})
    y = df["diagnosis_binary"].values
    return X, y

def train_test_split(x, y, train_size=0.7):
    data = int(x.shape[0] * train_size)
    return x[:data], x[data:], y[:data], y[data:]

def theta_x(theta, x):
    # Compute theta^T * x
    print("Shape of X:", x.shape)
    print("Shape of theta:", theta.shape)
    th_x = np.dot(x, theta)  # Ensure x is (m, n) and theta is (n,)
    return th_x

def sigmoid_func(z):
    # Compute the sigmoid function
    return 1 / (1 + np.exp(-z))

def main():
    # Load and prepare the data
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Initialize theta
    theta = np.zeros(x_train.shape[1])  # Shape (n,)

    # Add a bias column to x_train and x_test
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])  # Shape (m, n+1)
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])  # Shape (m, n+1)

    # Compute theta^T * x for the first batch
    z = theta_x(theta, x_train)  # Shape (m,)
    print("Theta^T * X (z):", z)

    # Apply the sigmoid function
    g = sigmoid_func(z)
    print("Sigmoid output:", g)


if __name__ == "__main__":
    main()

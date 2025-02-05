#Implement Stochastic Gradient Descent algorithm from scratch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def load_data():
    df = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = df.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = df["disease_score"].values
    y2 = df["disease_score_fluct"].values
    return X, y1, y2

def train_testsample_division(x,y):
    data = int(x.shape[0] * 0.70)
    # print(data)
    return (x[:data],x[data:],y[:data],y[data:])

def h_theta(x,th):
    h_theta=[]
    for i in range(x.shape[0]):
        h=0
        for j,k in zip(th,x[i]):
            h+=(j[0]*k)
        h_theta.append(h)
    return np.array(h_theta)

def Update_Params(th,alp,dervs):
    th_n=[]
    for i,j in zip(th,dervs):
        th_n.append([i[0]-alp*j])
    return np.array(th_n)


def cost_function(h,y):
    c_f=[]
    for x,y1 in zip(h,y):
        c_f.append((x-y1)**2)
    return (sum(c_f)/2)

def Derivative_CostF(x,y,h_theta):
    # print(x)
    x_t=[list(no) for no in (zip(*x))]
    # print(x_t)
    sum1=[]
    for i in x_t:
        sum2=0
        for j,k,l in zip(h_theta,y,i):
            sum2+=(j-k)*l
        sum1.append(np.clip(sum2, -1e10, 1e10))  # Clip gradient values
    return np.array(sum1)



def r_square_comp(x,y,th):
    y_m=np.mean(y,axis=0)
    h=h_theta(x,th)
    num=sum((i-j)**2 for i,j in zip(h,y))
    denom=sum((i-y_m)**2 for i in y)
    return 1-(num/denom)
    # return num/2


def gradient_descent(x_train, x_test, y_train, y_test, theta):
    # Normalize data
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[x_std == 0] = 1  # Avoid division by zero

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    # Add bias column
    new_col = np.ones((x_train.shape[0], 1))
    X_train = np.hstack((new_col, x_train))

    new_col2 = np.ones((x_test.shape[0], 1))
    X_test = np.hstack((new_col2, x_test))

    # Hyperparameters
    iteration = 1000
    alpha = 0.0001  # Lower learning rate
    cost_func = []

    for itr in range(iteration):
        h_t = h_theta(X_train, theta)  # Hypothesis
        c_f = cost_function(h_t, y_train)  # Cost function
        if c_f > 1e10:  # Divergence detection
            print(f"Gradient descent diverged at iteration {itr}")
            break
        cost_func.append(c_f)

        # Compute gradients
        der_cf = Derivative_CostF(X_train, y_train, h_t)

        # Update parameters
        theta = Update_Params(theta, alpha, der_cf)

        # Convergence check
        if itr > 0 and abs(cost_func[itr - 1] - cost_func[itr]) < 1e-6:
            print(f"Function converged at iteration {itr}")
            break

    print(f"Final theta values:\n{theta}\n")
    print(f"Final cost function:\n{cost_func[-1]}\n")

    # R^2 score
    r2 = r_square_comp(X_test, y_test, theta)
    print(f"R^2 score: {r2}")

    # Plot cost vs iteration
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(cost_func)), cost_func, color="blue", label="Cost per iteration")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function")
    plt.title("Cost vs iteration")
    plt.legend()
    plt.grid(True)
    plt.show()


def disease_gradient_descent(X, y1,thetas):
    # Train-test divide
    # for x in x.shape[0]
    X_Train, X_Test, Y_Train, Y_Test = train_testsample_division(X, y1)
    gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas)

def diseaseFluct_gradient_descent(X, y2,thetas):
    # Train-test divide
    X_Train, X_Test, Y_Train, Y_Test = train_testsample_division(X, y2)
    gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas)

def main():

    X, y1, y2 = load_data()
    thetas = np.zeros((X.shape[1] + 1, 1))
    print(thetas)


        # # FOR DISEASE COLUMN WITHOUT FLUCTUATION
    disease_gradient_descent(X, y1, thetas)

    print("\nDone!\n")

        # FOR DISEASE COLUMN WITH FLUCTUATION
    diseaseFluct_gradient_descent(X, y2, thetas)

    print("\nDone!\n")

if __name__ == "__main__":
    main()


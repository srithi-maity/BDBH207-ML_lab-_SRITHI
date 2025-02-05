import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    return X, y1


def Train_Test_Divide(x, y):
    sample_no= np.arange(x.shape[0])
    np.random.shuffle(sample_no)

    x = x[sample_no]
    y = y[sample_no]

    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]

def H_theta_calc(x,th):
    h_t_sum = []
    for i in range(x.shape[0]):
        h=0
        for j,k in zip(th, x[i]):
            h+=(j[0]*k)
        h_t_sum.append(h)
    return np.array(h_t_sum)

def cost_function(h,y):
    c_f=[]
    for x,y1 in zip(h,y):
        c_f.append((x-y1)**2)
    return (sum(c_f)/2)

def Derivative_CostF(x,y,h):
    x_t=[list(no) for no in (zip(*x))]
    sum1=[]
    for i in x_t:
        sum2=0
        for j,k,l in zip(h,y,i):
            sum2+=(j-k)*l
        sum1.append(sum2)
    return np.array(sum1)

def Update_Params(th,alp,dervs):
    th_n=[]
    for i,j in zip(th,dervs):
        th_n.append([i[0]-alp*j])
    return np.array(th_n)

def r_square_comp(x,y,th):
    y_m=np.mean(y,axis=0)
    h=H_theta_calc(x,th)
    num=sum((i-j)**2 for i,j in zip(h,y))
    denom=sum((i-y_m)**2 for i in y)
    return 1-(num/denom)

def gradient_descent(X_Train, X_Test, Y_Train, Y_Test,thetas):
    X_mean = np.mean(X_Train, axis=0)
    X_std = np.std(X_Train, axis=0)
    X_Train = (X_Train - X_mean) / X_std

    new_col = np.ones((X_Train.shape[0], 1))
    X_Train = np.hstack((new_col, X_Train))

    X_mean2 = np.mean(X_Test, axis=0)
    X_std2 = np.std(X_Test, axis=0)
    X_Test = (X_Test - X_mean2) / X_std2
    new_col2 = np.ones((X_Test.shape[0], 1))
    X_Test = np.hstack((new_col2, X_Test))


    iterations = 10000
    cost_funcs = []

    for iteration in range(iterations):
        # calculate hypothesis
        h_t = H_theta_calc(X_Train, thetas)

        # calculate cost function
        c_f = cost_function(h_t, Y_Train)
        cost_funcs.append(c_f)

        # calculate derivative of cost function
        der_cf = Derivative_CostF(X_Train, Y_Train, h_t)

        if iteration > 0 and abs(cost_funcs[iteration - 1] - cost_funcs[iteration]) < 1e-6:
            print(f"Function converged at iteration {iteration}\n")
            break

        # updating parameters
        alpha = 0.001
        thetas = Update_Params(thetas, alpha, der_cf)

    np.array(cost_funcs)
    print(f"Final theta values:\n{thetas}\n")
    print(f"Final cost function:\n{cost_funcs[-1]}\n")

    r2 = r_square_comp(X_Test, Y_Test, thetas)
    return r2



def main():

    x, y = load_data()

    k=3
    for f in range(k):
        x_train,x_test,y_train,y_test=Train_Test_Divide(x,y)
        print(f"mean of train_set for {f} fold is :{np.mean(x_train)}")
        theta = np.zeros((x_train.shape[1] + 1, 1))
        res= gradient_descent(x_train , x_test , y_train,y_test,theta)
        print(f" for the {f} no fold the r2 value is {res}")

    print("end")


if __name__ == "__main__":
    main()

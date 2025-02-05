import numpy as np
import pandas as pd
# from fontTools.ttLib.tables.otConverters import LTable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_data():
    # Load dataset
    df = pd.read_csv("/home/ibab/Downloads/archive/data.csv")
    df.fillna(0,inplace=True)
    x = df.drop(columns=["diagnosis","id","Unnamed: 32"]).values
    df["diagnosis_binary"]=df["diagnosis"].map({'M':1,'B':0})
    y = df["diagnosis_binary"].values
    return x,y

def split(x,y):

    X_train, X_test, y_train, y_test = train_test_split(x, y)
    # X_train = X_train.to_numpy()
    y_train = y_train.reshape(-1, 1)
    return X_train,X_test,y_train,y_test

def load():
    data = pd.read_csv("/home/ibab/Downloads/archive/data.csv")
    data.fillna(0,inplace=True)
    x=data.drop(columns=["diagnosis","id","Unnamed: 32"]).values
    data["diagnosis_binary"]=data["diagnosis"].map({'M':1,'B':0})
    y=data["diagnosis_binary"].values
    return x,y

# def train_test_split(x, y, train_size=0.7):
#     data = int(x.shape[0] * train_size)
#     return x[:data], x[data:], y[:data], y[data:]

def theta_x(th,x_train):
    th=th.reshape(-1,1)
    th_x=np.dot(x_train,th)
    return th_x

def sigmoid_func(z):
    g_th_t_x=1/(1+np.exp(-z))
    return g_th_t_x

# def derivative_sigmoid_func(x_train,th):
    # L_th_list=[]
    # L_th=1
    # for i in x_train :
    #     g_z=sigmoid_func(theta_x(th,x_train))
    #     der_g_z=g_z * (1-g_z)
    #     L_th*=der_g_z
    #     L_th_list.append(L_th)
    # return L_th_list

def derivative_sigmoid_func(g_z):
    der_g_z=g_z *(1-g_z)
    return der_g_z


def ScikitlearnLogisticReg():

    # Load data
    X, y = load_data()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Print classification metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

    # Print model parameters
    print(f"Theta values (coefficients): {model.coef_}")
    print(f"Bias term (intercept): {model.intercept_}")

def main():
    x,y=load_data()
    x1,x2,y1,y2= train_test_split(x, y, test_size=0.30, random_state=999)


    # Scale the data
    scaler = StandardScaler()
    scaler.fit(x1)
    x_train_scaled = scaler.transform(x1)
    x_test_scaled = scaler.transform(x2)
    new_col = np.ones((x_train_scaled.shape[0], 1))
    x_train_scaled = np.hstack((new_col, x_train_scaled))
    new_col2 = np.ones((x_test_scaled.shape[0], 1))
    x_test_scaled = np.hstack((new_col2, x_test_scaled))

    # x1,x2,y1,y2=split(x,y)
    x_train = np.hstack([np.ones((x1.shape[0], 1)), x1])  # Shape (m, n+1)
    theta = np.zeros(x_train.shape[1])
    z=theta_x(theta,x_train)
    g_z=sigmoid_func(z)
    # print(g_z)
    print(derivative_sigmoid_func(g_z))
    # print(derivative_sigmoid_func(x_train , theta))


if __name__ == "__main__":
    # ScikitlearnLogisticReg()
    main()
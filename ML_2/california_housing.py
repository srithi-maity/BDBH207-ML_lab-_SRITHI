from statistics import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# from sklearn import RandomForestClassifier


def load_data():
    [X,y]=fetch_california_housing(return_X_y=True)
    return (X,y)

def main():
    [X,y]=load_data()
    #split data - train=70% , test=30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)


    #scale the data
    scaler= StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #train a model
    print("__training__")

    model=LinearRegression()
    model.fit(X_train, y_train)#fit

    y_pred= model.predict(X_test) #prediction on a test set

    #compute the r2 score
    r2=r2_score(y_test, y_pred)
    print("r2 score is %0.2f (closer to 1 is good)" % r2)
    print("done!")


if __name__=="__main__":
    main()
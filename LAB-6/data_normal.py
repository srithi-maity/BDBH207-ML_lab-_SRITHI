import numpy as np
import pandas as pd



def load_data():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y = data["disease_score"].values
    return X, y

def main():
    x,y=load_data()
    for i in range (len(x)):
        for x_val in x[i]:

            x_min=min(x[i])
            # print(x_min)
            x_max=max(x[i])
            new_x=(x_val-x_min)/(x_max-x_min)
            print(new_x)



if __name__ == "__main__":
    main()

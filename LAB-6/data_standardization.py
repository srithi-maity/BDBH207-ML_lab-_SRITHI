import numpy as np
import pandas as pd



def load_data():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y = data["disease_score"].values
    return X, y

def main():
    x,y=load_data()
    x_cols=np.transpose(x)

    z=[]
    for i in range(len(x_cols)):
        mu=np.mean(x_cols[i])
        sigma=np.std(x_cols[i])
        z_inn=[]
        for x_val in x_cols[i]:
            z_score=(x_val-mu)/sigma
            z_inn.append(float(z_score))
        z.append(z_inn)
    print(z)
    # z_cols=np.transpose(z)
    # for j in range(len(z_cols)):
    #     meu=np.mean(z_cols[j])
    #     sig=np.std(z_cols[j])
    #     print(meu,sig)
    z_mean=np.mean(z,axis=0)
    print(z_mean[0])

if __name__ == "__main__":
    main()
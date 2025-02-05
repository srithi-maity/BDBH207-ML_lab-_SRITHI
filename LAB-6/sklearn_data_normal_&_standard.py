import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def main():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    print(f"scaled_data {scaled}")
    scaler_st=StandardScaler()
    standard_data=scaler_st.fit_transform(X)
    print(f"standard_data {standard_data}")


if __name__ == "__main__":
    main()
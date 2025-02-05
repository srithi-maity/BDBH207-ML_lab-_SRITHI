import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def load_data():
    # Load the dataset
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    print("Dataset Preview:")
    print(data.head(5))
    print(data.describe())



    # Split features and target variables
    # X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values

    y1 = data["disease_score"]
    y2 = data["disease_score_fluct"]  # Fixed column name
    return X, y1, y2,data


def main():
    # Load data
    X, y1, y2,data = load_data()




    # Split data - 70% train, 30% test
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.30, random_state=999)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Linear Regression model
    print("\nTraining Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y1_train)

    # Predict on test set
    y1_pred = model.predict(X_test_scaled)

    # Compute R-squared score
    r2 = r2_score(y1_test, y1_pred)
    print(f"R-squared score: {r2:.2f} (closer to 1 is better)")
    print("complete!")

    # Split data - 70% train, 30% test
    X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.30, random_state=999)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Linear Regression model
    print("\nTraining Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y2_train)

    # Predict on test set
    y2_pred = model.predict(X_test_scaled)

    # Compute R-squared score
    r2 = r2_score(y2_test, y2_pred)
    print(f"R-squared score: {r2:.2f} (closer to 1 is better)")
    print("complete!")

    # import seaborn as sns
    #
    # sns.scatterplot(
    #     data=data ,
    #     x="disease_score",
    #     y="BP",
    #     size="blood_sugar",
    #     hue="BMI",
    #     palette="viridis",
    #     alpha=0.5,
    # )
    # plt.legend(title="BP V/S disease_score", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    # plt.title("BP v/s disease_score")
    # plt.show()

    # import matplotlib.pyplot as plt

    # data.hist(figsize=(12, 10), bins=30, edgecolor="black")
    # plt.subplots_adjust(hspace=0.7, wspace=0.4)
    # plt.show()


    plt.scatter(data["age"], data["disease_score"], alpha=0.7, label="disease_score")
    plt.scatter(data["age"], data["disease_score_fluct"], alpha=0.7, label="disease_score_fluct")
    plt.xlabel("Age")
    plt.ylabel("Target")
    plt.legend()
    plt.title("Age vs Targets")
    plt.show()




if __name__ == "__main__":
    main()

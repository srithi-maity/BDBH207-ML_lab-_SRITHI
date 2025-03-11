import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot():
    data = pd.read_csv("/home/ibab/Downloads/DATA.csv", header=None)
    print("Original Data:")
    print(data.head())
    df = pd.DataFrame(data)

    # Automatically detect categorical columns (assuming non-numeric columns are categorical)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform categorical columns #####n machine learning, many models cannot handle categorical (non-numeric)
    # data directly.
    # Since categories are labels (like "Red", "Male", or "Yes"), they must be converted into numerical values.
    encoded_array = encoder.fit_transform(df[categorical_columns])

    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

    # Concatenate with original DataFrame if we don't do that they both will be seperted.
    df_encoded = pd.concat([df, encoded_df], axis=1)

    # Drop original categorical columns because we have made new so we will remove the previous version
    df_encoded.drop(categorical_columns, axis=1, inplace=True)

    print("Encoded Data:")
    print(df_encoded.head())  # Display first few rows

    print(f"Final Shape: {df_encoded.shape}")  # Print dimensions of final dataset


def main():
    one_hot()


if __name__ == "__main__":
    main()

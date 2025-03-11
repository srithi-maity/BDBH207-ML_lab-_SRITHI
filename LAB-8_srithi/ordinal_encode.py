# ordinal encode the breast cancer dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
def ordinal_encoding():

    url = "/home/ibab/Downloads/DATA.csv"
    dataset = read_csv(url, header=None)
    # retrieve the array of data
    data = dataset.values
    # separate into input and output columns
    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)
    # ordinal encode input variables
    ordinal_encoder = OrdinalEncoder()
    X = ordinal_encoder.fit_transform(X)
    # ordinal encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # summarize the transformed data
    print('Input', X.shape)
    print(X)
    # print(X[:5, :])
    print('Output', y.shape)
    print(y)
    # print(y[:5])

def main():
    ordinal_encoding()

if __name__=="__main__":
    main()
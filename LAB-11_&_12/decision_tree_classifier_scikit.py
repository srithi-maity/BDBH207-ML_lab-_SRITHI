import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix

## load the dataset
df=pd.read_csv("/home/ibab/Desktop/datasets/datasets/Iris.csv")
print(f"....EDA...of this dataset")
# print the first 5 columns
print(df.head(5))
print(df.info())
print(df.isnull().sum())

##pair plot
sns.pairplot(df,hue="Species")
# plt.show()

##box plot
plt.figure(figsize=(6,4))
sns.boxplot(x="SepalLengthCm",y="Species",data=df)
plt.title("plot of SepalLength vs Species")
# plt.show()

## Data_processing and defining the feature and target variables
df.drop(columns=["Id"],inplace=True)
x=df.drop(columns=["Species"])
y=df["Species"].astype("category").cat.codes
# y=df["Species"]
print(y.head())

## Train Test Splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)

## model
tree=DecisionTreeClassifier(max_depth=6,min_samples_split=2)
tree.fit(x_train,y_train)

# Making predictions
y_pred = tree.predict(x_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix plot
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plotting the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=df.columns[:-1], class_names=["Setosa", "Versicolor", "Virginica"], filled=True)
plt.show()



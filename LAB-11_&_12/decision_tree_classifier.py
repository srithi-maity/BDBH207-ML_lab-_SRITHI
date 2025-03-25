import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Loading the dataset
df = pd.read_csv("/home/ibab/Desktop/datasets/datasets/Iris.csv")
df.drop(columns=['Id'], inplace=True)
df['Species'] = df['Species'].astype('category').cat.codes

# Split dataset into features (X) and target (y)
X = df.drop(columns=['Species']).values
y = df['Species'].values


## here im seeing that if I use train test split then im getting 91 % accuracy and with the scratch one which
# I've commented there I'm getting 73% accuracy .
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.7,shuffle=True)
# up=int(X.shape[0]*0.7)
# X_train=X[:up]
# X_test=X[up:]
# y_train=y[:up]
# y_test =y[up:]

#  entropy calculation
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# information gain calculation
def information_gain(X_column, y, threshold):
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    parent_entropy = entropy(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    n = len(y)
    left_weight = len(y[left_mask]) / n
    right_weight = len(y[right_mask]) / n
    return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

# Building Decision Tree Node class
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Building Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])

        best_feature, best_threshold, best_gain = None, None, -1
        num_samples, num_features = X.shape
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

        if best_gain == 0:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _predict_sample(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(node.left, sample)
        return self._predict_sample(node.right, sample)

    def predict(self, X):
        return np.array([self._predict_sample(self.root, sample) for sample in X])

# Training the Decision Tree model
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)

# Making predictions
y_pred = tree.predict(X_test)

# Evaluating accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

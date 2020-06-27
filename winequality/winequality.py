import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# load data
data_wine_white = pd.read_csv("data/winequality-white.csv", sep=";")
data_wine_red = pd.read_csv("data/winequality-red.csv", sep=";")

# add type column
# white = 0, red = 1
data_wine_white["type"] = 0
data_wine_red["type"] = 1

# concat data frames
data = pd.concat([data_wine_white, data_wine_red])

# reset indices
data = data.reset_index()
del data["index"]

# separate x and y
feature_list = list(data)
feature_list.remove("type")
x = data[feature_list]
y = data[["type"]]

# Standardize inputs (for Principal Component Analysis, which I will explain later)
x = StandardScaler().fit_transform(x)

# split train and test data, allotting 40% of the data to test data.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=0)

'''
    For really big feature set, like MNIST dataset, who has 784 features, it will
take a really long time to fit a model for prediction using all of its features.
For that, we will perform Principal Component Analysis (or PCA) to find out what
features will take the most importance in determining the classification of wine.
    In PCA, we will find the features that has the largest variance ration in the
set of features. Then, we will use that features to train our model.
'''
# perform PCA, keeping only 6 of the principal components (not using all 12 features)
# since 6 components, constitute more than 80% of the variance of the data
# you can find out how many principal components you will use in this code:
# pca = PCA(n_components=12)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
pca = PCA(n_components=6)
principal_components = pca.fit_transform(train_x)

# find what are the principal components
n_components = 6
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_components)]
most_important_names = [feature_list[most_important[i]] for i in range(n_components)]
principal_component_names = {'PC{}'.format(i): most_important_names[i] for i in range(n_components)}

# the principal components are: total sulfur dioxide, density, citric acid, sulphates, residual sugar and pH
# now let's try to train a logistic regression model.
train_x_df = pd.DataFrame(train_x, columns=feature_list)
logistic_model = LogisticRegression()
logistic_model.fit(train_x_df[most_important_names], train_y)

# now let's predict from the test set.
test_x_df = pd.DataFrame(test_x, columns=feature_list)
y_hat = logistic_model.predict(test_x_df[most_important_names])

# compute confusion matrix
confusion_matrix = confusion_matrix(test_y, y_hat)
# Confusion matrix outputs TP, FP, TN, FN
# Results are:
# TP = 1957
# TN = 613
# FP = 13
# FN = 16
n_samples = sum(sum(confusion_matrix))
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / n_samples
sensitivity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
specificity = confusion_matrix[1, 1]/(confusion_matrix[1, 0] + confusion_matrix[1, 1])
print("Accuracy: " + str(accuracy))
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))

# Results:
# Accuracy: 0.9888418622547134
# Sensitivity: 0.9918905220476432
# Specificity: 0.9792332268370607

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# load data
data = load_breast_cancer(return_X_y=True, as_frame=True)
feature_list = list(data[0])
x = data[0]
y = data[1]

# Standardize inputs (for Principal Component Analysis, which I will explain later)
x = StandardScaler().fit_transform(x)

# split train and test data, allotting 40% of the data to test data.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=0)

'''
    For a really big feature set, like MNIST dataset, that has 784 features, it will
take a really long time to fit a model for prediction using all of its features.
    For this problem, we will perform Principal Component Analysis (or PCA) to find out what
features will take the most importance in determining the classification of breast cancer..
    In PCA, we will find the features that has the largest variance ration in the
set of features. Then, we will use that features to train our model.
'''
# perform PCA, keeping only 6 of the principal components (not using all 12 features)
# since 5 components, constitute more than 80% of the variance of the data
# you can find out how many principal components you will use in this code:
# pca = PCA(n_components=12)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
pca = PCA(n_components=5)
principal_components = pca.fit_transform(train_x)

# find what are the principal components
n_components = 5
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_components)]
most_important_names = [feature_list[most_important[i]] for i in range(n_components)]
principal_component_names = {'PC{}'.format(i): most_important_names[i] for i in range(n_components)}

# principal components are: mean concave points, mean fractal dimension, texture error, worst texture, concavity error
# now let's try to train a logistic regression model.
train_x_df = pd.DataFrame(train_x, columns=feature_list)
logistic_model = LogisticRegression()
logistic_model.fit(train_x_df[most_important_names], train_y)

# now let's predict from the test set.
test_x_df = pd.DataFrame(test_x, columns=feature_list)
y_hat = logistic_model.predict(test_x_df[most_important_names])

# compute confusion matrix
confusion_matrix = confusion_matrix(test_y, y_hat)
# Confusion matrix outputs [[TP, FN],[FP, TN]]
# Results are:
# TP = 72
# TN = 143
# FP = 2
# FN = 11
n_samples = sum(sum(confusion_matrix))
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / n_samples
sensitivity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
specificity = confusion_matrix[1, 1]/(confusion_matrix[1, 0] + confusion_matrix[1, 1])
print("Accuracy: " + str(accuracy))
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))

# Results:
# Accuracy: 0.9429824561403509
# Sensitivity: 0.8674698795180723
# Specificity: 0.9862068965517241


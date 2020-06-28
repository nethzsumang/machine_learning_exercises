import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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

# split train and test data, allotting 40% of the data to test data.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=0)

'''
    Now, we will try to fit a kNN model to predict its type. kNN is a classification algorithm wherein
the distance of the samples from each other are computed and it will determine where the specific 
sample is classified, i.e., if a sample is nearer to a white wine sample, then it is a white wine, else,
it is a red wine sample.
    In kNN algorithm, there is a constant called `k`. `k` is the number of neighbors needed to vote
that a sample is going into that category. In `k` = 3, it means that a sample must be near 3 white
wine samples to be considered to be a white wine sample.
    In R, we can have array of `k` values and a function will determine the best value of `k`.
But here in Python, we have to have a trial and error process to determine the best `k` value.
Low `k` values can result to overfitting and a high value of `k` can result to underfitting.
    For this problem, we will test our accuracy in a `k` range of 5 to 25.
'''
k_range = range(5, 26)
train_x_df = pd.DataFrame(train_x, columns=feature_list)
accuracies = {}

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_x_df, train_y)
    y_hat = knn_model.predict(train_x_df)
    accuracies[k] = metrics.accuracy_score(train_y, y_hat)

# now, we plot the values of the accuracies to their `k` value.
plt.plot(list(k_range), list(accuracies.values()))

# for this problem, we will choose 5 as `k`.
# now, to train our final model,
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_x_df, train_y)

# then, we predict from the test set
test_x_df = pd.DataFrame(test_x, columns=feature_list)
y_hat = knn_model.predict(test_x_df)

# we will now compute for the confusion matrix
confusion_matrix = metrics.confusion_matrix(test_y, y_hat)
# Confusion matrix outputs [[TP, FN],[FP, TN]]
# Results are:
# TP = 1916
# FP = 57
# TN = 521
# FN = 105
n_samples = sum(sum(confusion_matrix))
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / n_samples
sensitivity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
specificity = confusion_matrix[1, 1]/(confusion_matrix[1, 0] + confusion_matrix[1, 1])
print("Accuracy: " + str(accuracy))
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))

# Results:
# Accuracy: 0.9376683339746056
# Sensitivity: 0.9711099847947289
# Specificity: 0.8322683706070287

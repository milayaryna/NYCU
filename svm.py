import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = svm.SVC(kernel = 'rbf', C=1, gamma='auto')
model.fit(x_train, y_train)
model.predict(x_test)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

"""
TO DO:

what column has the best predictive power and how do I find this out?


"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

submission_csv = np.array([test_dataset.iloc[:, 0].values]).reshape(-1, 1)

# Dropping features that are currently not in use
train_dataset.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)
test_dataset.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)

# train_dataset.dropna(axis="index", how="any", subset=["Embarked"], inplace=True)
embarked_at = ["Q", "S", "C"]
train_dataset.fillna({"Embarked": random.choice(embarked_at)}, inplace=True) # Improve with weighted choice

# Splitting the data
X_train = train_dataset.iloc[:, 1:7].values

y_train = train_dataset.iloc[:, 0].values

X_test = test_dataset.iloc[:, :].values

# Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_train[:, 2] = imputer.fit_transform(X_train[:, 2].reshape(-1, 1)).ravel()

# Is it possible to get the right shape in the first place so as to not have to 
# reshape twice? copy=False??

X_test[:, 2] = imputer.transform(X_test[:, 2].reshape(-1, 1)).ravel()

# Encoding categorical data

labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder.transform(X_test[:, 1])

labelencoder_2 = LabelEncoder()
X_train[:, 5] = labelencoder_2.fit_transform(X_train[:, 5])
X_test[:, 5] = labelencoder_2.transform(X_test[:, 5])

onehotencoder = OneHotEncoder(categorical_features = [5])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]  # avoid Dummy Variable Trap

onehotencoder_2 = OneHotEncoder(categorical_features = [5])
X_test = onehotencoder_2.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]  # avoid Dummy Variable Trap

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 500)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, verbose = 10)

#mean = accuracies.mean()
#variance = accuracies.std()

classifier.fit(X_train, y_train)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Tuning the ANN

#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
#    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#
#classifier = KerasClassifier(build_fn = build_classifier)
#
#parameters = {'batch_size': [25, 32],
#              'epochs': [500],
#              'optimizer': ['adam', "rmsprop", "sgd"]}
#
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = 1,
#                           verbose = 10)
#
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_

# BEST PARAS FOUND: batch_size: 25, optimizier: rmsprop, epochs: 500

# Part 4 - Creating the submission csv file

submission_csv = np.hstack((submission_csv, y_pred))
submission_csv = pd.DataFrame(submission_csv)
submission_csv.to_csv("submission_4.csv", header=["PassengerId", "Survived"], index=False)








import mlrose_hiive as mlrose
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import make_scorer, plot_confusion_matrix, log_loss
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv("train_2.csv")
data_test = pd.read_csv("test_2.csv")

both_df = pd.concat([data, data_test], axis=0).reset_index(drop=True)

both_df.drop(["subject"], axis=1, inplace=True)
both_df = shuffle(both_df)


std_scaler = StandardScaler()
copy_both_df = both_df.copy()
X = both_df.loc[:, both_df.columns != "Activity"]
std_scaler.fit(X)
X_scaled = std_scaler.fit_transform(X)
y = both_df.Activity


y_encode = LabelEncoder().fit_transform(y)
labels = preprocessing.LabelEncoder().fit(y).classes_

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encode, random_state=2, stratify=y_encode)
iters = 400
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid', \
                                 algorithm = 'genetic_alg', max_iters = iters, \
                                 bias = True, is_classifier = True, learning_rate = 0.001, \
                                 early_stopping = False, clip_max = 5, max_attempts = iters, \
                                 random_state = 2, pop_size=200)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid', \
#                                  algorithm = 'gradient_descent', max_iters = iters, \
#                                  bias = True, is_classifier = True, learning_rate = 0.001, \
#                                  early_stopping = False, clip_max = 5, max_attempts = iters, \
#                                  random_state = 2)
#
# schedule = mlrose.ExpDecay(exp_const=.000777111)
# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid', \
#                                  algorithm = 'simulated_annealing', max_iters = iters, \
#                                  bias = True, is_classifier = True, learning_rate = 0.001, \
#                                  early_stopping = False, clip_max = 5, max_attempts = iters, \
#                                  random_state = 2, schedule=schedule)

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid', \
#                                  algorithm = 'random_hill_climb', max_iters = iters, \
#                                  bias = True, is_classifier = True, learning_rate = 0.001, \
#                                  early_stopping = False, clip_max = 5, max_attempts = iters, \
#                                  random_state = 2, restarts=5)
one_hot = OneHotEncoder()

y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).todense()
y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).todense()
t1_train = datetime.datetime.now()
nn_model1.fit(X_train, y_train)
t2_train = datetime.datetime.now()
t_train = t2_train - t1_train
print(t_train)

metric_scores_train = nn_model1.predict(X_train)
print(precision_score(y_train, metric_scores_train, average="weighted"))
#
t1_train = datetime.datetime.now()
metric_scores_test = nn_model1.predict(X_test)

t2_train = datetime.datetime.now()
t_train = t2_train - t1_train
print(t_train)
print(precision_score(y_test, metric_scores_test, average="weighted"))
#y_train_pred = nn_model1.predict(X_train)
print("hi")

# print(len(nn_model1.fitted_weights))
# print(schedule.evaluate(len(nn_model1.fitness_curve)))
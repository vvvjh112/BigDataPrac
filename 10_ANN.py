import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.preprocessing import *
from matplotlib import pyplot as plt
from sklearn.neural_network import *
from scipy.stats import *

train = pd.read_csv('dataset/PimaIndiansDiabetes2.csv')

print(train.columns)

train['diabetes'] = LabelEncoder().fit_transform(train['diabetes'])

train = train.dropna()

x = train.drop('diabetes', axis = 1)
y = train['diabetes']

model = MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1000)

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

model.fit(trainX,trainY)

pred = model.predict(testX)

print(recall_score(testY,pred))
print(accuracy_score(testY,pred))
print(precision_score(testY,pred))
print(f1_score(testY,pred))
print(roc_auc_score(testY,pred))
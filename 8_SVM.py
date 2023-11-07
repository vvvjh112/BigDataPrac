import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.preprocessing import *

train = pd.read_csv('dataset/PimaIndiansDiabetes2.csv')

print(train.columns)

train['diabetes'] = LabelEncoder().fit_transform(train['diabetes'])

train = train.dropna()

print(train.isna().sum())
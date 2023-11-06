import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

data = np.array([1,3,5,7,9])
x = data.reshape(-1,1)
minimax = MinMaxScaler().fit_transform(x)
print(minimax)
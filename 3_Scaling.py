import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from datetime import datetime

# 최소 최대 정규화 방법1
from sklearn.preprocessing import MinMaxScaler

data = np.array([1,3,5,7,9])
x = data.reshape(-1,1)
minimax = MinMaxScaler().fit_transform(x)
print(minimax)

#방법2
data = np.array([1,3,5,7,9])

def minimax_Scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

print(minimax_Scale(data))

#표준화 Z점수
data = np.array([1,3,5,7,9])

#ddof는 자유도
def z_score(x):
    return (x - np.mean(x))/np.std(x,ddof=1)

data_z = z_score(data)
print(np.mean(data_z))
print(np.std(data_z,ddof=1))

print(data_z)

#표본 추출
ary = np.arange(1,11)
tmp = np.random.choice(ary, size=5, replace=False)
#replace는 복원추출이냐 비복원추출이냐 True 복원추출
print(tmp)
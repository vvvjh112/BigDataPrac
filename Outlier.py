import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ESD를 이용한 이상값 검출(3표준편차 이내의 값들이면 True 반환)
def esd(x):
    return abs((x-x.mean())/x.std())<3

print(train[esd(train['hour_bef_temperature'])])
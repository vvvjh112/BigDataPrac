import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print(train.isna().sum()) #컬럼별 결측값 개수 카운팅

print("결측값 삭제(행기준)-------")

train = train.dropna(axis=0)

print(train.isna().sum())


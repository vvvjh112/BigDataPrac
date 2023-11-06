import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print(train.isna().sum()) #컬럼별 결측값 개수 카운팅

test1 = train.std() #결측값에 대한 처리 이전의 표준편차

# print("결측값 삭제(행기준)-------")

# train = train.dropna(axis=0)

# print(train.isna().sum())

print("결측값 평균으로 대치")

train = train.fillna(train.mean())

print(train.isna().sum())

#중위수로 대체할 경우, median

# train = train.fillna(train.median())

#결측값에 대한 처리를 진행한 후 표준편차
test2 = train.std()

#특정 컬럼에 대해서도 확인 가능
print(test1['hour_bef_temperature'] - test2['hour_bef_temperature'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr


#MSE(Mean Squared Error) 평균제곱오차
#RMSE -> MSE의 양의 제곱근
#값이 낮을수록 모형의 정확도가 높다.
from sklearn.metrics import mean_squared_error
y_true = [3,5,7,9]
y_pred = [2,5,8,10]

print(mean_squared_error(y_true,y_pred,squared=True)) # True면 MSE False면 RMSE
print(mean_squared_error(y_true,y_pred,squared=False)) # True면 MSE False면 RMSE

#결정계수
#선형 회귀분석의 성능 검증지표로 많이 이용 ( 선형이 아니더라도 다른 회귀모형에서 사용가능)
#실제값을 얼마나 잘 나타내는지에 대한 비율 1에 가까울수록 잘 설명
from sklearn.metrics import r2_score
print(r2_score(y_true,y_pred))

#혼동행렬
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

y_true = [1,0,1,0,1]
y_pred = [1,1,1,0,0]

confusion_ary = confusion_matrix(y_true,y_pred)
print(confusion_ary)
recall = recall_score(y_true,y_pred)
accuracy = accuracy_score(y_true,y_pred)
precision = precision_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
print(recall, accuracy, precision,f1)

#ROC곡선
#0.5 ~ 1의 값을 가지며 1에 가까울수록 좋은 모형
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_true,y_pred))
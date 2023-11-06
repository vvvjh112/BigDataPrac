import pandas as pd
import numpy as np

mtcars = pd.read_csv('dataset/mtcars.csv')
#mpg - 연비, cyl - 엔진의 기통 수, wt - 차의 무게, am - 변속기어(0 : 자동, 1 : 수동)

#범주형 데이터에 대하여 빈도수 탐색
print(pd.Series(mtcars['cyl']).value_counts())

#수치형 데이터 탐색
print(mtcars['wt'].describe())

#다차원 데이터 탐색
#범주형-범주형 데이터 탐색
#빈도수와 비율을 활용
print(pd.crosstab(mtcars['am'],mtcars['cyl']))

#수치형 - 수치형 데이터 탐색
#상관계수를 활용하여 탐색 (수치형은 피어슨계수)
print(mtcars['mpg'].corr(mtcars['wt'],'pearson'))
# 'spearman', kendall
# 스피어만의 경우는 두 변수가 순서적 데이터일 경우
# 켄달의 경우 두 변수가 순서적 데이터일 경우 계량적으로 산출

df = pd.read_csv('dataset/PimaIndiansDiabetes.csv')
df = df.iloc[:,[2,3,4,7]]
df = df.dropna()
print(df.describe())
print(df.corr(method='pearson'))

#범주형-수치형
#범주형 데이터의 항목들을 그룹으로 간주하고 항목들에 관한 기술 통계량으로 데이터를 탐색한다.
#groupby사용
print(mtcars.groupby('cyl')['mpg'].mean())
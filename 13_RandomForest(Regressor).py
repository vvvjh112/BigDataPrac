import pandas as pd
from scipy.stats import *

data = pd.DataFrame({'before':[5,3,8,4,3,2,1],'after':[8,6,6,5,8,7,3]})

result = ttest_rel(data['before'],data['after'],alternative = 'less')

print(result)

#TtestResult(statistic=-2.633628675421043, pvalue=0.019435182851729293, df=6)
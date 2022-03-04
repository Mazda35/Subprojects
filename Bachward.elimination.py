

import numpy as np
import pandas as pd
import matplotlib.pyplot as pld

# Read data from dataset
dataset=pd.read_csv('50_Startups.csv')
print('\n\nDataset is:-----------------------\n',dataset)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print('\n\nX is:-----------------------\n',X)
print('\n\nY is:-----------------------\n',Y)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoderX=LabelEncoder()
X[:,-1]=labelEncoderX.fit_transform(X[:,-1])
print('\n\nCategorical 1 X is:-----------------------\n',X)
# oneHotEncoder=OneHotEncoder(categories=X[:,0])
# X=oneHotEncoder.fit_transform(X)
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
print('\n\nCategorical 2 X is:-----------------------\n',X)
X=X[:,1:]
print('\n\nCategorical 1 X is:-----------------------\n',X)

import statsmodels.formula.api as sm


significantVariables=[0,1,2,3,4,5]
insignificantVariables=[]
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
while True:
    print('\nVariables at start:',significantVariables)
    # print('\n\nX is:-----------------------\n', X)
    xOptimall = X[:,significantVariables]
    # print(X,'\t',xOptimall)

    xOptimall = np.array(xOptimall, dtype=float)
    # print('\n\nX optimal is:-----------------------\n', xOptimal)

    regressorOLS = sm.OLS(endog=Y, exog=xOptimall).fit()
    pValues = list(regressorOLS.pvalues)
    m=max(pValues)
    # print('\n\np value is:-----------------------\n', pValues)
    if m>0.05: # in case you need another significance level you can change
        insignificantVariables.append(significantVariables[pValues.index(m)])
        significantVariables.pop(pValues.index(m))
    else:
        break
print('\nSignificant Variables are:-----------------------\n',significantVariables)
print('\nInsignificant Variables are:-----------------------\n',insignificantVariables)

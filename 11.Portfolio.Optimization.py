
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl

# Get Data by Quandl
startDate=pd.to_datetime('2012-01-01')
endDate=pd.to_datetime('2017-01-01')

# apple=quandl.get('WIKI/AAPL',start_date=startDate,end_date=endDate)
# apple.to_csv('11.Apple.csv')
# cisco=quandl.get('WIKI/CSCO',start_date=startDate,end_date=endDate)
# cisco.to_csv('11.Cisco.csv')
# ibm=quandl.get('WIKI/IBM',start_date=startDate,end_date=endDate)
# ibm.to_csv('11.IBM.csv')
# amzn=quandl.get('WIKI/AMZN',start_date=startDate,end_date=endDate)
# amzn.to_csv('11.Amazon.csv')

# Get Data from hard
apple=pd.read_csv('11.Apple.csv')
apple['Date']=pd.to_datetime(apple['Date'])
apple.set_index('Date',inplace=True)
ibm=pd.read_csv('11.IBM.csv')
ibm['Date']=pd.to_datetime(ibm['Date'])
ibm.set_index('Date',inplace=True)
cisco=pd.read_csv('11.Cisco.csv')
cisco['Date']=pd.to_datetime(cisco['Date'])
cisco.set_index('Date',inplace=True)
amazon=pd.read_csv('11.Amazon.csv')
amazon['Date']=pd.to_datetime(amazon['Date'])
amazon.set_index('Date',inplace=True)


allstocks=[apple['Adj. Close'],cisco['Adj. Close'],ibm['Adj. Close'],amazon['Adj. Close']]
portfolio=pd.concat(allstocks,axis=1)
portfolio.columns=['Apple','Cisco','IBM','Amazon']
print(portfolio)
covariance=portfolio.cov()
print('Covariance\n',covariance)
m=max(covariance)
print(m)


print('\n\n\n\n')
# Mean Daily Return & Correlation
print(portfolio.pct_change(1).mean())
print(portfolio.pct_change(1).corr())

returnArit=portfolio.pct_change(1)
print(returnArit.head())
returnLog=np.log(portfolio/portfolio.shift(1))
print(returnLog.head())

# returnLog.hist(bins=100)
# plt.tight_layout()
# plt.show()

# print(returnLog.mean())
# print(returnLog.cov())
# print(returnLog.corr())

# Weights
# np.random.seed(101)
# weights=np.array(np.random.random(4))
# print('Random Weights: ',weights)
# weights=weights/np.sum(weights)
# print('Normal Weights: ',weights)

# Expected Return & Volatility
# expectedReturn=np.sum((returnLog.mean()*weights)*252)
# print('Expected Return: ',expectedReturn)
# expectedVolatility=np.sqrt(np.dot(weights.T,np.dot(returnLog.cov()*252,weights)))
# print('Expected Volatility: ',expectedVolatility)
# SR=expectedReturn/expectedVolatility
# print('Sharpe Ratio: ',SR)

np.random.seed(101)

numberPortfolios=5000
allWeights=np.zeros((numberPortfolios,len(portfolio.columns)))
returnsArray=np.zeros(numberPortfolios)
volatilityArray=np.zeros(numberPortfolios)
sharpeRatioArray=np.zeros(numberPortfolios)


for index in range(numberPortfolios):
    weights=np.array(np.random.random(4))
    weights=weights/np.sum(weights)

    allWeights[index,:]=weights
    returnsArray[index]=np.sum((returnLog.mean()*weights)*252)
    volatilityArray[index]=np.sqrt(np.dot(weights.T,np.dot(returnLog.cov()*252,weights)))
    sharpeRatioArray[index]=returnsArray[index]/volatilityArray[index]

print(sharpeRatioArray.max())
print(sharpeRatioArray.argmax())
print(allWeights[sharpeRatioArray.argmax(),:])

maxReturn=returnsArray[sharpeRatioArray.argmax()]
maxVolatility=volatilityArray[sharpeRatioArray.argmax()]

# Plotting
# plt.figure()
plt.scatter(volatilityArray,returnsArray,c=sharpeRatioArray,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(maxVolatility,maxReturn,c='Green')
plt.show()


def getReturnVolatilitySharpeRatio(weights):
    weights=np.array(weights)
    returnsArray=np.sum((returnLog.mean() * weights) * 252)
    volatilityArray=np.sqrt(np.dot(weights.T, np.dot(returnLog.cov() * 252, weights)))
    sharpeRatioArray= returnsArray/ volatilityArray
    return np.array([returnsArray,volatilityArray,sharpeRatioArray])

from scipy.optimize import minimize
def negativeSharpeRatio(weights):
    return getReturnVolatilitySharpeRatio(weights)[2]*(-1)

def checkSum(weights):
#     return 0 if the sum of weights is 1
    return np.sum(weights)-1

constraints=({'type':'eq','fun':checkSum})
bounds=((0,1),(0,1),(0,1),(0,1))
initialGuess=[0.25,0.25,0.25,0.25]
optimalResults=minimize(negativeSharpeRatio,initialGuess,method='SLSQP',
                        bounds=bounds,constraints=constraints)
print(optimalResults)

# Efficient Frontier
def minimizeVolatility(weights):
    return getReturnVolatilitySharpeRatio(weights)[1]

frontierVolatility=[]
frontierY=np.linspace(0,0.3,100)

for possibleReturn in frontierY:
    constraints=({'type':'eq','fun':checkSum},
                 {'type':'eq','fun':lambda w:getReturnVolatilitySharpeRatio(w)[0]-possibleReturn})
    result=minimize(minimizeVolatility,initialGuess,method='SLSQP',bounds=bounds,constraints=constraints)
    frontierVolatility.append(result['fun'])

plt.scatter(volatilityArray,returnsArray,c=sharpeRatioArray,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontierVolatility,frontierY,'g')
plt.show()
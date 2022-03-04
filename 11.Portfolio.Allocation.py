###############################################################################################
# Portfolio Allocation
###############################################################################################

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

allStocks=[apple,cisco,ibm,amazon]
for stock in allStocks:
    stock['Normal Return']=stock['Adj. Close']/stock['Adj. Close'].iloc[0]
    del stock['Open'],stock['Close'],stock['Low'],stock['High'],stock['Volume']
    del stock['Ex-Dividend'],stock['Split Ratio']
    del stock['Adj. Open'],stock['Adj. High'],stock['Adj. Low'],stock['Adj. Volume']

# Aloocation (0.3, 0.2,0.4,0.1)

for stock,allo in zip(allStocks,[0.3,0.2,0.4,0.1]):
    stock['Allocation']=stock['Normal Return']*allo

for stock in allStocks:
    stock['Position']=stock['Allocation']*1000000
    stock['Position']=stock['Position'].round(0)

# print(amazon)

allPositionValue=[apple['Position'],cisco['Position'],ibm['Position'],amazon['Position']]
portfolio=pd.concat(allPositionValue,axis=1)
portfolio.columns=['Apple Position','Cisco Position','IBM Position','Amazon Position']
portfolio['Value']=portfolio.sum(axis=1)

portfolio['Value'].plot(figsize=(8,8))
plt.title('Portfolio Value')
plt.show()

portfolio.drop('Value',axis=1).plot(figsize=(8,8))
# portfolio.plot(figsize=(8,8))
plt.legend()
plt.title('Portfolio Value vs Stocks')
plt.show()

##################################################################################### Part2
portfolio['Daily Return']=portfolio['Value'].pct_change(1)
print(portfolio)
print(portfolio['Daily Return'].mean())
print(portfolio['Daily Return'].std())

portfolio['Daily Return'].plot(kind='hist',figsize=(6,4),bins=100)
portfolio['Daily Return'].plot(kind='kde',figsize=(6,4),color='red')
plt.show()

# Cumulative Return
cumulativeReturn=100*(portfolio['Value'][-1]/portfolio['Value'][0]-1)
print('Cumulative Return: ',cumulativeReturn.round(3))

# Sharpe Ratio=(Rp-Rf)/stdf
Rp=portfolio['Daily Return'].mean()
Rf=0/252
stdf=portfolio['Daily Return'].std()
SR=(Rp-Rf)/stdf
print('Sarpe Ratio: ',SR)

annualSR=252**0.5*SR
print('Annual SR: ',annualSR)


























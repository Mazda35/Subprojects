
import pandas as pd
import numpy as np
import pandas_datareader as web
import scipy.stats as sst
import datetime
import matplotlib.pyplot as plt

start=pd.to_datetime('2010-01-04')
end=pd.to_datetime('2017-07-25')

# spy=web.DataReader('SPY','yahoo',start=start,end=end)
# apple=web.DataReader('AAPL','yahoo',start=start,end=end)
# spy.to_csv('11..Spy.csv')
# apple.to_csv('11..Apple.csv')

spy=pd.read_csv('11..Spy.csv')
spy['Date']=pd.to_datetime(spy['Date'])
spy.set_index('Date',inplace=True)
apple=pd.read_csv('11..Apple.csv')
apple['Date']=pd.to_datetime(apple['Date'])
apple.set_index('Date',inplace=True)
# print(spy.info())

apple['Close']=apple['Close'].mul(4)

apple['Close'].plot(label='Apple')
spy['Close'].plot(label='Spy')
plt.title('Apple vs Market')
plt.legend()
plt.show()

apple['Cumulative']=apple['Close']/apple['Close'].iloc[0]
spy['Cumulative']=spy['Close']/spy['Close'].iloc[0]

apple['Cumulative'].plot(label='Apple')
spy['Cumulative'].plot(label='Spy')
plt.title('Apple vs Market (Cumulative)')
plt.legend()
plt.show()

apple['Daily Return']=apple['Close'].pct_change(1)
spy['Daily Return']=spy['Close'].pct_change(1)

plt.scatter(apple['Daily Return'],spy['Daily Return'],alpha=0.25)
plt.show()
# plt.scatter(apple['Close'],spy['Close'],alpha=0.25)
# plt.show()

beta,alpha,r_value,p_value,std_err=sst.linregress(apple['Daily Return'].iloc[1:],spy['Daily Return'].iloc[1:])
print('Beta: ',beta,'\nAlpha: ',alpha,'\nR value :',r_value,'\nP value: ',p_value,'\nStandard Deviation: ',std_err)

# beta,alpha,r_value,p_value,std_err=sst.linregress(spy['Daily Return'].iloc[1:],apple['Daily Return'].iloc[1:])
# print(beta,alpha,r_value,p_value,std_err)

noise=np.random.normal(0,0.001,len(spy['Daily Return'].iloc[1:]))
fake=spy['Daily Return'].iloc[1:]+noise

plt.scatter(fake,spy['Daily Return'].iloc[1:],alpha=0.25)
plt.show()

beta,alpha,r_value,p_value,std_err=sst.linregress(fake,spy['Daily Return'].iloc[1:])
print('Beta: ',beta,'\nAlpha: ',alpha,'\nR value :',r_value,'\nP value: ',p_value,'\nStandard Deviation: ',std_err)


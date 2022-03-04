
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import mpl_finance as pl
from matplotlib.dates import DateFormatter,date2num,WeekdayLocator,MONDAY,DateLocator


ford=pd.read_csv('Ford_Stock.csv')
gm=pd.read_csv('GM_Stock.csv')
tesla=pd.read_csv('Tesla_Stock.csv')

ford['Date']=pd.to_datetime(ford['Date'])
ford.set_index('Date',inplace=True)
gm['Date']=pd.to_datetime(gm['Date'])
gm.set_index('Date',inplace=True)
tesla['Date']=pd.to_datetime(tesla['Date'])
tesla.set_index('Date',inplace=True)

# print('Ford: ...................\n',ford)
# print('GM: ...................\n',gm)
# print('Tesla ...................\n:',tesla)

# Plot Open
ford['Open'].plot(figsize=(24,6),color='blue',label='Ford')
gm['Open'].plot(color='green',label='GM')
tesla['Open'].plot(color='purple',label='Tesla')
plt.legend()
plt.title('Stocks Open')
plt.show()

# Close
ford['Close'].plot(figsize=(24,6),color='blue',label='Ford')
gm['Close'].plot(color='green',label='GM')
tesla['Close'].plot(color='purple',label='Tesla')
plt.legend()
plt.title('Stocks Close')
plt.show()

# Volume
ford['Volume'].plot(figsize=(24,6),color='blue',label='Ford')
gm['Volume'].plot(color='green',label='GM')
tesla['Volume'].plot(color='purple',label='Tesla')
plt.legend()
plt.title('Stocks Volume')
plt.show()

ford['Total Trade']=ford['Open'].mul(ford['Volume']).round(0)
gm['Total Trade']=gm['Open'].mul(gm['Volume']).round(0)
tesla['Total Trade']=tesla['Open'].mul(tesla['Volume']).round(0)

print('Ford: ...................\n',ford)
print('GM: ...................\n',gm)
print('Tesla ...................\n:',tesla)

# Plot Total Trade
ford['Total Trade'].plot(figsize=(24,6),color='blue',label='Ford')
gm['Total Trade'].plot(color='green',label='GM')
tesla['Total Trade'].plot(color='purple',label='Tesla')
plt.legend()
plt.title('Stocks Total Trade')
plt.show()

# Moving Average
ford['Open'].plot(figsize=(24,8),color='blue',label='Open')
ford['Open'].rolling(50).mean().plot(color='orange',label='Move 50')
ford['Open'].rolling(200).mean().plot(color='green',label='Move 50')
plt.legend()
plt.title('Moving Average Ford')
plt.show()

# Scatter plot
df=pd.concat([ford['Open'],gm['Open'],tesla['Open']],axis=1)
df.columns=['Ford Open','GM Open','Tesla Open']
scatter_matrix(df)
plt.show()

# Candle Sticks
# fordJanuary=ford.loc['2012-01'].reset_index()
# print(fordJanuary)
# pl.candlestick2_ochl(ax=1,opens=fordJanuary['Open'],closes=fordJanuary['Close'],
#                      highs=fordJanuary['High'],lows=fordJanuary['Low'])
# plt.show()

# Returns
tesla['Return']=tesla['Close'].pct_change(1)
ford['Return']=ford['Close'].pct_change(1)
gm['Return']=gm['Close'].pct_change(1)

# Histogram
tesla['Return'].hist(bins=100,label='Tesla')
ford['Return'].hist(bins=100,label='Ford')
gm['Return'].hist(bins=100,label='GM')
plt.legend()
plt.show()

# KDE
tesla['Return'].plot(kind='kde',label='Tesla')
gm['Return'].plot(kind='kde',label='GM')
ford['Return'].plot(kind='kde',label='Ford')
plt.legend()
plt.show()

# Box Plot
dfBox=pd.concat([tesla['Return'],ford['Return'],gm['Return']],axis=1)
dfBox.columns=['Tesla Return','Ford Return','GM Return']
dfBox.plot(kind='box')
plt.title('Box Plot')
plt.show()

scatter_matrix(dfBox,hist_kwds={'bins':100})
plt.show()

dfBox.plot(kind='scatter',x='Ford Return',y='GM Return',alpha=0.5)
plt.show()

print(dfBox['Ford Return'].corr(dfBox['GM Return']))
print(dfBox['Ford Return'].corr(dfBox['Tesla Return']))
print(dfBox['Tesla Return'].corr(dfBox['GM Return']))

# Cumulative Return
tesla['Cumulative Return']=(1+tesla['Return']).cumprod()
ford['Cumulative Return']=(1+ford['Return']).cumprod()
gm['Cumulative Return']=(1+gm['Return']).cumprod()

tesla['Cumulative Return'].plot(label='Tesla')
gm['Cumulative Return'].plot(label='GM')
ford['Cumulative Return'].plot(label='Ford')
plt.legend()
plt.show()












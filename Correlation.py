

#Correlation
globals()['Samsung'] = DataReader('005930.KS','yahoo',start,end)
globals()['AAPL'] = DataReader('AAPL','yahoo',start,end)
globals()['MSFT'] = DataReader('MSFT','yahoo',start,end)
CP = DataReader(['005930.KS','AAPL','MSFT'], 'yahoo', start, end)['Close']
pct_cg = CP.pct_change()
pct_cg=pct_cg.dropna()
pct_cg.head()

AAPL['Close'].plot(subplots=False,figsize=(50,20))
MSFT['Close'].plot(subplots=False,figsize=(50,20))
Samsung['Close'].plot(subplots=False,figsize=(50,20))

# Comparing Samsung to itself will show a perfectly positive correlaton
sns.jointplot('005930.KS','005930.KS',pct_cg,kind='scatter',color='orange')

# We'll use joinplot to compare the daily returns of Apple and MS.
sns.jointplot('AAPL','MSFT',pct_cg, kind='scatter',size=8, color='skyblue')

# We can simply call pairplot ans use it to correlate every feature
sns.pairplot(pct_cg,size=3)

# Lets check out the correlation between closing prices of stocks
sns.heatmap(pct_cg.corr(),annot=True,fmt=".3g",cmap='YlGnBu')

# Lets check out the correlation between closing prices of stocks
sns.heatmap(CP.corr(),annot=True,fmt=".3g",cmap='YlGnBu')

#Risk Analysis Bootstrap Method
sns.distplot(pct_cg['005930.KS'],bins=100)

pct_cg['005930.KS'].quantile(0.01)
pct_cg['MSFT'].quantile(0.01)
pct_cg['AAPL'].quantile(0.01)

#Monte Carlo Simulation
#for MSFT
days = 260
last_price = CP['MSFT'][-1]
mu = pct_cg.mean()['MSFT']

def monteCarlo(start_price,days,mu):
    price = np.zeros(days)
    price[0] = start_price
    for x in range(1,days):
        price[x] = price[x-1]*(1+np.random.normal(0,mu))     
    return price

for run in range(500):
    plt.plot(monteCarlo(last_price, days, mu))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for MSFT')
plt.figure(figsize=(50,20))

#for AAPL
last_price = CP['AAPL'][-1]
mu = pct_cg.mean()['AAPL']

for run in range(500):
    plt.plot(monteCarlo(last_price, days, mu))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for AAPL')

#for Samsumg
last_price = CP['005930.KS'][-1]
mu = pct_cg.mean()['005930.KS']

for run in range(500):
    plt.plot(monteCarlo(last_price, days, mu))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Samsung')

# Lets Dive Deeper
# For MSFT Stocks
days = 260
last_price = CP['MSFT'][-1]
print(last_price)
# number of simulations
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = monteCarlo(last_price,days,mu)[days-1]

sns.distplot(simulations,bins=100)
sm = DataFrame(simulations)
sm.quantile(0.01)

# for AAPL Stocks
days = 260
last_price = CP['AAPL'][-1]
print(last_price)
# number of simulations
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = monteCarlo(last_price,days,mu)[days-1]

sns.distplot(simulations,bins=100)
sm = DataFrame(simulations)
sm.quantile(0.01)


# for Samsung Stock
last_price = CP['005930.KS'][-1]
print(last_price)
# number of simulations
runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = monteCarlo(last_price,days,mu)[days-1]


sns.distplot(simulations,bins=100)
sm = DataFrame(simulations)
sm.quantile(0.01)


#Selecting Best Features

Features = NFLX[['KAMA','CCI','ADL','WilliamsR','STCD','STCK','BullPower','BearPower','FI','CMO','ATR','RSI','MACD','pctB','TEMA','UBB','LBB','EMA of 22 Days','MA for 22 days','ChandelierExitL','Close']]
plt.figure(figsize = (100,100))
sns.heatmap(Features.corr(),annot=True,fmt=".3g",cmap='YlGnBu')
Features.corr()['Close'].sort_values()
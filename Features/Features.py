# For Data Processing
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

# Data Visualization
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline


# For reading stock data from yahoo
NFLX = pd.read_csv('NFLX.csv')

#ploting all the columns of My Dataset 
plt.figure()
plt.subplot(2,2,1)
NFLX['Close'].plot(legend=True, figsize=(10,4))
plt.subplot(222)
NFLX['Open'].plot(legend=True, figsize=(10,4))
plt.subplot(223)
NFLX['Volume'].plot(legend=True, figsize=(10,4))
plt.subplot(224)
NFLX['Adj Close'].plot(legend=True, figsize=(10,4))

#1 Moving Average(MA)
MA_day = [10,22]
for ma in MA_day:
    column_name = 'MA for %s days' %(str(ma))
    NFLX[column_name] = NFLX['Close'].rolling(ma).mean()  #pd.rolling_mean(AAPL['Close'],ma)
    
NFLX[['Close','MA for 10 days','MA for 22 days']].plot(subplots=False,figsize=(50,20))

#2 Exponential Moving Average (EMA)
emaPeriod = 22
NFLX['EMA of 22 Days'] = NFLX['Close'].ewm(com=emaPeriod -1,min_periods =emaPeriod).mean()
NFLX[['Close','MA for 22 days','EMA of 22 Days']].plot(subplots=False,figsize=(50,20))

#3 Bollinger Bands (BB)
SD = NFLX['Close'].rolling(22).std()
NFLX['UBB'] = NFLX['MA for 22 days'] + (2*SD)
NFLX['LBB'] = NFLX['MA for 22 days'] - (2*SD)
NFLX[['Close','MA for 22 days','UBB','LBB']].plot(subplots=False,figsize=(50,20))

#4 Triple Exponential Moving Average (TEMA)
NFLX['EMA2'] = NFLX['EMA of 22 Days'].ewm(com=emaPeriod -1,min_periods =emaPeriod).mean()
NFLX['EMA3'] = NFLX['EMA2'].ewm(com=emaPeriod -1,min_periods =emaPeriod).mean()
NFLX['TEMA'] = 3*NFLX['EMA of 22 Days'] - 3*NFLX['EMA2'] + NFLX['EMA3']
NFLX[['Close','EMA of 22 Days','TEMA']].plot(subplots=False,figsize=(50,20))

#5 %b
NFLX['pctB'] = 20 * (NFLX['Close'] - NFLX['LBB']) / (NFLX['UBB'] - NFLX['LBB'])
NFLX[['Close','pctB','MA for 22 days','UBB','LBB']].plot(subplots=False,figsize=(50,20))

#6 Moving Average Convergence/Divergence (MACD)
MA12 = NFLX['Close'].rolling(12).mean()
MA26 = NFLX['Close'].rolling(26).mean()
MACD = MA12 - MA26
NFLX['MACD'] = MACD
NFLX[['Close','MACD']].plot(subplots=False,figsize=(50,20))

#7 Relative Strength Index (RSI)
rsiVal=14
diffrence = NFLX['Close'].diff()
gain = diffrence.mask(diffrence < 0, 0) 
loss = diffrence.mask(diffrence > 0, 0) 
avgGain = gain.ewm(com=rsiVal-1 ,min_periods=rsiVal).mean()
avgLoss = loss.ewm(com=rsiVal-1 ,min_periods=rsiVal).mean()
rs = abs(avgGain/avgLoss)
rsi = 100*(rs/(1+rs))
NFLX['RSI'] = rsi
NFLX[['Close','RSI']].plot(subplots=False,figsize=(50,20))

#8 Average True Range (ATR) : measure of Volitality
def TRange(h,l,yc):
    x=h-l
    y=abs(h-yc)
    z=abs(l-yc)
    if y <= x >= z:
        TR = x
    elif x <= y >=z:
        TR = y
    elif x <= z >=y:
        TR = z
    return TR
x=1
TrueRange = []
TrueRange.append(0)
while x < len(NFLX['Close']):
    TR = TRange(NFLX['High'][x],NFLX['Low'][x],NFLX['Close'][x-1])
    TrueRange.append(TR)
    x = x+1
NFLX['TR'] = TrueRange
NFLX['ATR'] = NFLX['TR'].ewm(com=emaPeriod -1,min_periods =emaPeriod).mean()
NFLX[['Close','TR','ATR']].plot(subplots=False,figsize=(50,20))

#9  Chandelier Exit 
def Highest(n,emaPeriod):
    max=NFLX['Close'][n]
    for i in range(n,n+emaPeriod+1):
        if NFLX['Close'][i] > max:
            max = NFLX['Close'][i]
    return max
fac =3
ChandelierExitL=np.array([])
ChandelierExitL = np.append(ChandelierExitL, [np.nan]*emaPeriod)
for i in range(0,len(NFLX['Close'])-emaPeriod):
    ChandelierExit = Highest(i,emaPeriod) - fac*NFLX['ATR'][i+emaPeriod]
    ChandelierExitL = np.append(ChandelierExitL,ChandelierExit)
NFLX['ChandelierExitL']=ChandelierExitL
NFLX[['Close','ChandelierExitL']].plot(subplots=False,figsize=(50,20))

#10 Chande Momentum Oscillator (MOM)
def cmo(timeFrame):
    x=timeFrame
    CMO = []
    for i in range(0,timeFrame):
        CMO.append(0);
    NFLX['Diff'] = NFLX['Close'].diff()
    while x < len(NFLX['Close']):
        upSum=0
        downSum=0
        y=x-timeFrame+1
        while y<x:
            if NFLX['Diff'][y]>=0:
                upSum += NFLX['Diff'][y]
            else:
                downSum += NFLX['Diff'][y]
            y+=1
        curCMO = ((upSum-downSum)/(upSum+float(downSum)))*100
        CMO.append(curCMO)
        x+=1
    NFLX['CMO'] = CMO
    return
cmo(10)
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['CMO'].plot(subplots=False,figsize=(50,20))

#11 Force Index (FI)
FI = np.array([])
FI = np.append(FI, np.nan)
for i in range(1,len(NFLX['Close'])):
    FIVal = (NFLX['Close'][i] - NFLX['Close'][i-1])*NFLX['Volume'][i]
    FI = np.append(FI,FIVal)
NFLX['FI']=FI
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['FI'].plot(subplots=False,figsize=(50,20))

#12 Elder-ray
BullPower = np.array([])
BullPower = np.append(BullPower, [np.nan]*emaPeriod)
BearPower = np.array([])
BearPower = np.append(BearPower, [np.nan]*emaPeriod)
for i in range(emaPeriod,len(NFLX['Close'])):
    BullP = NFLX['High'][i] - NFLX['EMA of 22 Days'][i]
    BullPower = np.append(BullPower,BullP)
    BearP = NFLX['Low'][i] - NFLX['EMA of 22 Days'][i]
    BearPower = np.append(BearPower,BearP)
NFLX['BullPower'] = BullPower
NFLX['BearPower'] = BearPower
plt.figure()
plt.subplot(3,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(3,1,2)
NFLX['BullPower'].plot(subplots=False,figsize=(50,20))
plt.subplot(3,1,3)
NFLX['BearPower'].plot(subplots=False,figsize=(50,20))

#13 Stochastic %k (STCK)
def HighestLowest(n,period):
    high=NFLX['High'][n]
    low=NFLX['Low'][n]
    for i in range(n,n+period+1):
        if NFLX['High'][i] > high:
            high = NFLX['High'][i]
        if NFLX['Low'][i] < low:
            low = NFLX['Low'][i]
    return high,low
STCK = np.array([])
STCK = np.append(STCK, [np.nan]*emaPeriod)
for i in range(0,len(NFLX['Close'])-emaPeriod):
    high,low=HighestLowest(i,emaPeriod)
    PK = 100*(NFLX['Close'][i+emaPeriod] - low)/(high - low)
    STCK = np.append(STCK, PK)
NFLX['STCK'] = STCK
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['STCK'].plot(subplots=False,figsize=(50,20))

#14 Stochastic %D (STCD)
NFLX['STCD'] = NFLX['STCK'].ewm(com = 2 , min_periods = 3).mean()
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['STCD'].plot(subplots=False,figsize=(50,20))

#15 Williams %R
def Highest(n,period):
    high=NFLX['High'][n]
    for i in range(n,n+period+1):
        if NFLX['High'][i] > high:
            high = NFLX['High'][i]
    return high
def Lowest(n,period):
    low=NFLX['Low'][n]
    for i in range(n,period+1):
        if NFLX['Low'][i] < low:
            low = NFLX['Low'][i]
    return low
WilliamsR = np.array([])
WilliamsR = np.append(WilliamsR, [np.nan]*emaPeriod)
for i in range(0,len(NFLX['Close'])-emaPeriod):
    high=Highest(i,emaPeriod)
    low=Lowest(i,emaPeriod)
    W = ((-100)*(high - NFLX['Close'][i+emaPeriod]))/((high - float(low)))
    WilliamsR = np.append(WilliamsR, W)
NFLX['WilliamsR'] = WilliamsR
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['WilliamsR'].plot(subplots=False,figsize=(50,20))

#16  Accumulation Distribution Oscillation (ADO)
NFLX['Multiplier']= (2*NFLX['Close']-NFLX['High']-NFLX['Low'])/(NFLX['High']-NFLX['Low'])
NFLX['MVolume']=NFLX['Multiplier']*NFLX['Volume']
ADL = np.array([])
ADL = np.append(ADL,NFLX['MVolume'][0])
for i in range(1,len(NFLX['Close'])):
    Adl = ADL[i-1] + NFLX['MVolume'][i]
    ADL = np.append(ADL,Adl)
NFLX['ADL'] = ADL
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['ADL'].plot(subplots=False,figsize=(50,20))

#17 Commodity Channel Index (CCI)
def mad(n,period):
    sum=0
    for i in range(n,n+period+1):
        sum+=abs(NFLX['TP']-NFLX['SMP of TP'])
    return high
NFLX['TP'] = (NFLX['Close']+NFLX['High']+NFLX['Low'])/3
NFLX['SMP of TP'] = NFLX['TP'].rolling(emaPeriod).mean()
MAD = np.array([])
MAD = np.append(MAD,[np.nan]*emaPeriod)
for i in range(0,len(NFLX['Close'])-emaPeriod):
    su = mad(i,emaPeriod)
    MAD = np.append(MAD,su/float(emaPeriod))
NFLX['MAD of TP'] = MAD
NFLX['CCI'] = (NFLX['TP'] - NFLX['SMP of TP'])/(0.015*NFLX['MAD of TP'])
plt.figure()
plt.subplot(2,1,1)
NFLX['Close'].plot(subplots=False,figsize=(50,20))
plt.subplot(2,1,2)
NFLX['CCI'].plot(subplots=False,figsize=(50,20))

#18 Kaufman's Adaptive Moving Average (KAMA)
KAMA = np.array([])
KAMA = np.append(KAMA,[np.nan]*9)
KAMA = np.append(KAMA,NFLX['Close'][9])
Volitality = np.array([])
Volitality = np.append(Volitality, np.nan)
for i in range(1,len(NFLX['Close'])):
    Volitality = np.append(Volitality, abs(NFLX['Close'][i]-NFLX['Close'][i-1]))
Change = np.array([])
Change = np.append(Change,[np.nan]*10)
for i in range(10,len(NFLX['Close'])):
    Change = np.append(Change, abs(NFLX['Close'][i]-NFLX['Close'][i-10]))
ER = np.array([])
ER = np.append(ER, [np.nan]*10)
for i in range(1,len(NFLX['Close'])-9):
    sum=0;
    for j in range(i,i+9):
        sum+=Volitality[j]
    Val=Change[i+9]/sum
    ER = np.append(ER,Val)
SC = np.array([])
SC = np.append(SC, [np.nan]*10)
for i in range(10,len(NFLX['Close'])):
    slowest = 2.0/3.0
    fastest = 2.0/31.0
    S = pow((ER[i]*(fastest - slowest)+slowest),2)
    SC = np.append(SC, S)
for i in range(10,len(NFLX['Close'])):
    K = KAMA[i-1] + (NFLX['Close'][i]-KAMA[i-1])*SC[i]
    KAMA = np.append(KAMA,K)
NFLX['KAMA'] = KAMA
NFLX[['Close','KAMA']].plot(subplots=False,figsize=(50,20))


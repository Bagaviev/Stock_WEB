import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', 50)
#1================================================================================================================================
import datetime as dt
actual = dt.datetime.now()
stop_dt = dt.datetime.now().date()
start_dt_Y = actual.date() - dt.timedelta(days=365)
start_dt_M = actual.date() - dt.timedelta(days=30)
start_dt_W = actual.date() - dt.timedelta(days=7)

import pandas_datareader.data as web
sample = web.DataReader('EURUSD000TOM', 'moex', start=str(start_dt_Y), end=str(actual.date()))
sample.reset_index(inplace=True,drop=False)
sample = sample[sample['BOARDID'] == 'CETS']

da = sample['TRADEDATE'].max().date()
data = dt.datetime.combine(da, dt.datetime.min.time())
diff = actual - data

def actuator():
    if (diff.days < 1):
        sample['is_actual'] = 'Today'
    elif(diff.days == 1):
        sample['is_actual'] = '1 day lag'
    elif(diff.days > 1 & diff.days < 3):
        sample['is_actual'] = 'Couple days lag'
    else:
        sample['is_actual'] = 'Outdated'

actuator()

agg = sample[['CLOSE', 'TRADEDATE']].sort_values(by=['TRADEDATE'], ascending=False).head(1)
sample['last_CLOSE'] = agg.CLOSE
sample.reset_index(inplace=True)
sample = sample[['TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLRUR', 'is_actual', 'last_CLOSE']]
#2================================================================================================================================
import matplotlib.pyplot as plt
plt.figure(figsize=(15,4))

smoothing = 4
smooth_prices = sample['CLOSE'].rolling(window=smoothing).mean().dropna()

import numpy as np
from scipy.signal import argrelextrema

local_max = argrelextrema(smooth_prices.values, np.greater)[0]
local_min = argrelextrema(smooth_prices.values, np.less)[0]

window_range=3

price_local_max_dt = []
for i in local_max:
    if (i>window_range) and (i<len(sample)-window_range):
        price_local_max_dt.append(sample.iloc[i-window_range:i+window_range]['CLOSE'].idxmax())
        
price_local_min_dt = []
for i in local_min:
    if (i>window_range) and (i<len(sample)-window_range):
        price_local_min_dt.append(sample.iloc[i-window_range:i+window_range]['CLOSE'].idxmin())
        
maxima = pd.DataFrame(sample.loc[price_local_max_dt])
minima = pd.DataFrame(sample.loc[price_local_min_dt])
max_min = pd.concat([maxima, minima]).sort_index()

max_min.index.name = 'date'
max_min = max_min.reset_index()
max_min = max_min[~max_min.date.duplicated()]
p = sample.reset_index()


from collections import defaultdict

def find_patterns(max_min):
    patterns = defaultdict(list)

    for i in range(5, len(max_min)):
        window = max_min['CLOSE'].iloc[i-5:i]

        # pattern must play out in less than 36 days
#         if window.index[-1] - window.index[0] > 10:
#             continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        rtop_g1 = np.mean([e1,e3,e5])
        rtop_g2 = np.mean([e2,e4])
        # Head and Shoulders
        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
            (abs(e1 - e5) <= 0.03*np.mean([e1,e5])) and \
            (abs(e2 - e4) <= 0.03*np.mean([e1,e5])):
                patterns['HS'].append((window.index[0], window.index[-1]))

        # Inverse Head and Shoulders
        elif (e1 < e2) and (e3 < e1) and (e3 < e5) and \
            (abs(e1 - e5) <= 0.03*np.mean([e1,e5])) and \
            (abs(e2 - e4) <= 0.03*np.mean([e1,e5])):
                patterns['IHS'].append((window.index[0], window.index[-1]))

        # Broadening Top
        elif (e1 > e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['BTOP'].append((window.index[0], window.index[-1]))

        # Broadening Bottom
        elif (e1 < e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['BBOT'].append((window.index[0], window.index[-1]))

        # Triangle Top
        elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['TTOP'].append((window.index[0], window.index[-1]))

        # Triangle Bottom
        elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['TBOT'].append((window.index[0], window.index[-1]))

        # Rectangle Top
        elif (e1 > e2) and (abs(e1-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e3-rtop_g1)/rtop_g1 < 0.0075) and (abs(e5-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e2-rtop_g2)/rtop_g2 < 0.0075) and (abs(e4-rtop_g2)/rtop_g2 < 0.0075) and \
            (min(e1, e3, e5) > max(e2, e4)):

            patterns['RTOP'].append((window.index[0], window.index[-1]))

        # Rectangle Bottom
        elif (e1 < e2) and (abs(e1-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e3-rtop_g1)/rtop_g1 < 0.0075) and (abs(e5-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e2-rtop_g2)/rtop_g2 < 0.0075) and (abs(e4-rtop_g2)/rtop_g2 < 0.0075) and \
            (max(e1, e3, e5) > min(e2, e4)):
            patterns['RBOT'].append((window.index[0], window.index[-1]))
            
    return patterns 

patterns2 = find_patterns(max_min)

d = list(patterns2.values())
RBOT = []
HS = []
TTOP = []
BTOP = []
IHS = []
TBOT = []
RTOP = []
for i, j in enumerate(d):
    if i == 0:
        for q in j:
            for p in q:
                RBOT.append(p)
    if i == 1:
        for q in j:
            for p in q:
                HS.append(p)
    if i == 2:
        for q in j:
            for p in q:
                TTOP.append(p)
    if i == 3:        
        for q in j:
            for p in q:
                BTOP.append(p)
    if i == 4:
        for q in j:
            for p in q:
                IHS.append(p)
    if i == 5:
        for q in j:
            for p in q:
                TBOT.append(p)
    if i == 6:
        for q in j:
            for p in q:
                RTOP.append(p)
                
RBOT1 = pd.DataFrame(max_min.loc[RBOT])
HS1 = pd.DataFrame(max_min.loc[HS])
TTOP1 = pd.DataFrame(max_min.loc[TTOP])
BTOP1 = pd.DataFrame(max_min.loc[BTOP])
IHS1 = pd.DataFrame(max_min.loc[IHS])
TBOT1 = pd.DataFrame(max_min.loc[TBOT])
RTOP1 = pd.DataFrame(max_min.loc[RTOP])

for i in RBOT:
    RBOT1 = RBOT1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in HS:
    HS1 = HS1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in TTOP:
    TTOP1 = TTOP1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in BTOP:
    BTOP1 = BTOP1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in IHS:
    IHS1 = IHS1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in TBOT:
    TBOT1 = TBOT1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
for i in RTOP:
    RTOP1 = RTOP1.append(max_min[['date','CLOSE']].loc[max_min['date'] == i])
#3================================================================================================================================
import talib as ta
import time
sample['TRADEDATE'] = pd.to_datetime(sample['TRADEDATE'], dayfirst=True)
sample['TRADEDATE'] = sample['TRADEDATE'].dropna().astype(str)

sample['short'] = sample['CLOSE'].ewm(span=3, adjust=False).mean().dropna()
sample['long'] = sample['CLOSE'].ewm(span=9, adjust=False).mean().dropna()
sample['diff'] = sample['short'] - sample['long']
X = 0.0035
sample['Stance'] = np.where(sample['diff'] > X, 1, 0)
sample['Stance'] = np.where(sample['diff'] < -X, -1, sample['Stance'])
sample['OBV'] = ta.OBV(sample['CLOSE'], sample['VOLRUR'])
sample['STHST_k'], sample['STHST_d'] = ta.STOCH(sample['HIGH'], sample['LOW'], sample['CLOSE'], fastk_period=14, slowk_period=2, slowk_matype=0, slowd_period=3, slowd_matype=0)
sample['BBU'], sample['BBM'], sample['BBL'] = ta.BBANDS(sample['CLOSE'], timeperiod=20, nbdevup=2)

sample['diff_st'] = sample['STHST_k'] - sample['STHST_d']
Y = 1
sample['Stance_st'] = np.where(sample['diff_st'] > Y, 1, 0)
sample['Stance_st'] = np.where(sample['diff_st'] < -Y, -1, sample['Stance_st'])

def conditions(x):
    if x > 0:
        return "Покупать (Long)"
    elif x == 0:
        return "Ожидать (Hold)"
    else:
        return "Продавать (Short)"

func = np.vectorize(conditions)
sample["Recommendation"] = func(sample["Stance_st"])
title = "Total recommendations for '{}' to trade: {}".format(sample.TRADEDATE.tail(1).to_string(index=False).replace(' ', ''), sample.Recommendation.tail(1).to_string(index=False).replace(' ', ''))
#4================================================================================================================================
year = sample
month = sample[sample['TRADEDATE'] >= str(start_dt_M)]
week = sample[sample['TRADEDATE'] >= str(start_dt_W)]

path = "D:\Desktop_D\УЧЕБА\MAGA_IU5\sem4\diploma_shit\Exchange_Web\static\\"

def plotter(period, period_name):
    import matplotlib as mpl
    import numpy as np
    
    recommendation = period.tail(3).plot(ylim=[-1.1,1.1], x='TRADEDATE', y='Stance_st', figsize=(5, 2), grid=True, title=title)
    
    ax1 = period.plot(grid=True, x='TRADEDATE', y='CLOSE', linewidth=1.8, figsize=(13, 5))    
    ax2 = period.plot(grid=True, x='TRADEDATE', y='BBU', color='gray', linestyle=':', linewidth=2, ax=ax1)    
    ax3 = period.plot(grid=True, x='TRADEDATE', y='BBL', color='gray', linestyle=':', linewidth=2, ax=ax1)
    ax4 = period.plot(grid=True, x='TRADEDATE', y='short', linewidth=1.5, color='orange', linestyle='-.', ax=ax1)
    ax5 = period.plot(grid=True, x='TRADEDATE', y='long', linewidth=1.5, color='m', linestyle='-.', ax=ax1)
    
    stoch = period[['STHST_k','STHST_d', 'TRADEDATE']].plot(x='TRADEDATE', figsize=(13, 5), grid=True)
    obv = period.plot(x='TRADEDATE', y='OBV', figsize=(13, 5), grid=True)
    
    if (period_name == 'y'):
        plt.figure(figsize=(13,5))
        year.reset_index().plot(grid=True, x='TRADEDATE', y='CLOSE', figsize=(13, 5))
        plt.scatter(RBOT1.date, RBOT1['CLOSE'].values, s=25, color='orange', alpha=1)
        plt.savefig('{}rbot_y.png'.format(path), bbox_inches='tight')
    elif (period_name == 'm'):
        plt.figure(figsize=(13,5))
        year.reset_index().plot(grid=True, x='TRADEDATE', y='CLOSE', figsize=(13, 5))
        plt.scatter(TTOP1.date, TTOP1['CLOSE'].values, s=25, color='blue', alpha=1)
        plt.savefig('{}ttop_y.png'.format(path), bbox_inches='tight')        
    elif (period_name == 'w'):
        plt.figure(figsize=(13,5))
        year.reset_index().plot(grid=True, x='TRADEDATE', y='CLOSE', figsize=(13, 5))
        plt.scatter(HS1.date, HS1['CLOSE'].values, s=25, color='red', alpha=1)
        plt.savefig('{}hs_y.png'.format(path), bbox_inches='tight')
    else:
        print('Error')
    
    recommendation.get_figure().savefig('{}recommendation.png'.format(path), bbox_inches='tight')
    ax1.get_figure().savefig('{}main_{}.png'.format(path, period_name), bbox_inches='tight')
    stoch.get_figure().savefig('{}stoch_{}.png'.format(path, period_name), bbox_inches='tight')
    obv.get_figure().savefig('{}obv_{}.png'.format(path, period_name), bbox_inches='tight')
    
plotter(year, 'y')
plotter(month, 'm')
plotter(week, 'w')

year.to_csv('D:\Desktop_D\УЧЕБА\MAGA_IU5\sem4\diploma_shit\Exchange_Web\static\data\parsed_EURUSD_y.csv', sep=';', encoding='utf-8')
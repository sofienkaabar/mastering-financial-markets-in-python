import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
# run this block of code if you want to remove warnings (aesthetic reasons only)
import warnings
warnings.filterwarnings("ignore")
'''

def generate_ohlc_data(length_data=1000):
    # create zero-value ohlc arrays
    data = {
        'open': np.zeros(length_data),
        'high': np.zeros(length_data),
        'low': np.zeros(length_data),
        'close': np.zeros(length_data)}
    # initialize the first prices
    data['open'][0]  = np.random.uniform(100, 200)
    data['close'][0] = data['open'][0] + np.random.uniform(-5, 5)
    data['high'][0]  = max(data['open'][0], data['close'][0]) + np.random.uniform(0, 5)
    data['low'][0]   = min(data['open'][0], data['close'][0]) - np.random.uniform(0, 5)
    # simulate the path of the hypothetical time series
    for i in range(1, length_data):
        data['open'][i]  = data['close'][i-1] + np.random.uniform(-3, 3)
        data['close'][i] = data['open'][i] + np.random.uniform(-5, 5)
        data['high'][i]  = max(data['open'][i], data['close'][i]) + np.random.uniform(0, 5)
        data['low'][i]   = min(data['open'][i], data['close'][i]) - np.random.uniform(0, 5)
    # convert to pandas dataframe
    my_time_series = pd.DataFrame(data)
    return my_time_series

def import_data(name='NVDA', start_date='2017-01-01', end_date='2025-10-04', data_provider='yahoo_finance', time_frame='daily'):
    if data_provider == 'yahoo_finance': # FX and stocks
        import yfinance as yf
        my_time_series = yf.download(name, start=start_date, end=end_date)
        my_time_series = my_time_series.xs(name, level=1, axis=1) 
        del my_time_series['Volume']
        my_time_series.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
        my_time_series = my_time_series[['open', 'high', 'low', 'close']]
    elif data_provider == 'metatrader': # FX
        import MetaTrader5 as mt5
        import datetime
        import pytz
        def get_quotes(time_frame=time_frame, year=2005, month=1, day=1, name=name):    
            if not mt5.initialize(): 
                print('initialize() failed, error code =', mt5.last_error()) 
                quit()
            timezone = pytz.timezone('Europe/Paris') # change to fit your timezone
            time_from = datetime.datetime(year, month, day, tzinfo=timezone)
            time_to = datetime.datetime.now(timezone) + datetime.timedelta(days=1)
            rates = mt5.copy_rates_range(name, time_frame, time_from, time_to)
            rates_frame = pd.DataFrame(rates)
            return rates_frame
        if time_frame == '15_minute':
            my_time_series = get_quotes(mt5.TIMEFRAME_M15, 2025, 8, 1, name) # put date close to today 
        if time_frame == '30_minute':
            my_time_series = get_quotes(mt5.TIMEFRAME_M30, 2025, 8, 1, name) # put date close to today 
        if time_frame == 'hourly':
            my_time_series = get_quotes(mt5.TIMEFRAME_H1, 2022, 1, 1, name)
        if time_frame == 'daily':
            my_time_series = get_quotes(mt5.TIMEFRAME_D1, 2010, 1, 1, name)
        if time_frame == 'weekly':
            my_time_series = get_quotes(mt5.TIMEFRAME_W1, 2007, 1, 1, name)
        if time_frame == 'monthly':
            my_time_series = get_quotes(mt5.TIMEFRAME_MN1, 2005, 1, 1, name)    
        my_time_series['Date'] = pd.to_datetime(my_time_series['time'], unit='s')
        my_time_series = my_time_series.reindex(columns=['Date', 'open', 'high', 'low', 'close'])
        my_time_series = my_time_series.set_index('Date')        
    elif data_provider == 'fred': # economic data
        import pandas_datareader as pdr
        my_time_series = pdr.get_data_fred(name, start=start_date, end=end_date).dropna()
        my_time_series.columns.values[0] = 'close'   
    elif data_provider == 'manual_import_xlsx': # anydata in xlsx format
        my_time_series = pd.read_excel(name)
        my_time_series['Date'] = pd.to_datetime(my_time_series['Date'], format='%m/%d/%Y')
        my_time_series = my_time_series.set_index('Date')
    elif data_provider == 'manual_import_csv': # anydata in csv format
        my_time_series = pd.read_csv(name)
        my_time_series['Date'] = pd.to_datetime(my_time_series['Date'], format='%m/%d/%Y')
        my_time_series = my_time_series.set_index('Date')
    else:
        print('Invalid import method, use yahoo_finance, metatrader, fred, manual_import_xlsx, or manual_import_csv')
    return my_time_series

def performance_evaluation(my_time_series, strategy='variable_holding_period', holding_period=10, theoretical_equity_curve=False):
    '''
    choosing strategy=variable_holding_period will result in signals ending upon encountering another signal
    choosing strategy=fixed_holding_period will result in signals ending after holding_period number of rows
    setting theoretical_equity_curve=True will plot an equity curve of the strategy based on profit/loss gross values
    '''
    my_time_series['bullish_result'] = 0
    my_time_series['bearish_result'] = 0
    if strategy == 'fixed_holding_period':
        for i in range(len(my_time_series)):
            if my_time_series['bullish_signal'].iloc[i]==1:
                if i+holding_period < len(my_time_series):
                    my_time_series['bullish_result'].iloc[i+holding_period] = my_time_series['close'].iloc[i+holding_period] - \
                                                                    my_time_series['open'].iloc[i]
            if my_time_series['bearish_signal'].iloc[i]==1:
                if i+holding_period < len(my_time_series):
                    my_time_series['bearish_result'].iloc[i+holding_period] = my_time_series['open'].iloc[i] - \
                                                                    my_time_series['close'].iloc[i+holding_period]
        bullish_hits = (my_time_series['bullish_result']>0).sum()
        bearish_hits = (my_time_series['bearish_result']>0).sum()
        total_bullish_trades = (my_time_series['bullish_signal']!=0).sum()
        total_bearish_trades = (my_time_series['bearish_signal']!=0).sum()
        total_hit_ratio = (bullish_hits + bearish_hits) / (total_bullish_trades + total_bearish_trades)
    elif strategy == 'variable_holding_period':
        for i in range(len(my_time_series)): 
            if my_time_series['bullish_signal'].iloc[i]==1:
                for j in range(i+1, len(my_time_series)):
                    if my_time_series['bearish_signal'].iloc[j]==1 or my_time_series['bullish_signal'].iloc[j]==1:
                        my_time_series['bullish_result'].iloc[j] = my_time_series['open'].iloc[j] - my_time_series['open'].iloc[i]
                        break
                    else:
                        continue
            if my_time_series['bearish_signal'].iloc[i]==1:
                for j in range(i+1, len(my_time_series)):
                    if my_time_series['bullish_signal'].iloc[j]==1 or my_time_series['bearish_signal'].iloc[j]==1:
                        my_time_series['bearish_result'].iloc[j] = my_time_series['open'].iloc[i] - my_time_series['open'].iloc[j]
                        break
                    else:
                        continue                    
    bullish_hits = (my_time_series['bullish_result']>0).sum()
    bearish_hits = (my_time_series['bearish_result']>0).sum()
    bullish_misses = (my_time_series['bullish_result']<0).sum()
    bearish_misses = (my_time_series['bearish_result']<0).sum()
    total_bullish_trades = (my_time_series['bullish_signal']!=0).sum()
    total_bearish_trades = (my_time_series['bearish_signal']!=0).sum()
    bullish_hit_ratio = bullish_hits / total_bullish_trades
    bearish_hit_ratio = bearish_hits / total_bearish_trades
    total_hit_ratio = (bullish_hits + bearish_hits) / (total_bullish_trades + total_bearish_trades)
    profit_factor = (bullish_hits + bearish_hits) / (bullish_misses + bearish_misses)
    combined = pd.concat([my_time_series['bullish_result'], my_time_series['bearish_result']])
    gains = combined[combined > 0]
    losses = combined[combined < 0].abs()
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss != 0:
        risk_reward_ratio = avg_gain / avg_loss
    else:
        risk_reward_ratio = np.inf
    expectancy = (avg_gain * total_hit_ratio) - (avg_loss * (1-total_hit_ratio))
    gains_losses = pd.concat([gains, losses])
    average_result = gains_losses.mean()
    standard_deviation = gains_losses.std()
    sharpe = average_result / standard_deviation
    downside_deviation = losses.std()
    sortino = average_result / downside_deviation
    print('')        
    print('---Performance Evaluation---')
    print('Total hit ratio = ', round(total_hit_ratio*100, 2), '%')
    print('Bullish hit ratio = ', round(bullish_hit_ratio*100, 2), '%')
    print('Bearish hit ratio = ', round(bearish_hit_ratio*100, 2), '%')
    print('Profit factor = ', round(profit_factor, 2))
    print('Risk-reward ratio = ', round(risk_reward_ratio, 2))    
    print('Expectancy = ', round(expectancy, 2))
    print('Sharpe ratio = ', round(sharpe, 2))     
    print('Sortino ratio = ', round(sortino, 2))        
    print('Number of bullish signals = ', total_bullish_trades)    
    print('Number of bearish signals = ', total_bearish_trades)  
    print('Number of signals = ', total_bullish_trades + total_bearish_trades)
    print('------')
    if theoretical_equity_curve == True:
        my_time_series['total_result'] = 0
        for i in range(len(my_time_series)):
            my_time_series['total_result'].iloc[i] = my_time_series['total_result'].iloc[i-1] + \
                                                     my_time_series['bullish_result'].iloc[i] + \
                                                     my_time_series['bearish_result'].iloc[i]
        fig, ax = plt.subplots()
        ax.plot(my_time_series['total_result'], label='Theoretical Equity Curve')
        ax.legend()
        ax.grid()
        ax.axhline(y=0, color = 'black', linestyle='dashed')
        plt.tight_layout()
    return my_time_series
    
def ohlc_plot(my_time_series, window=250, plot_type='bars', chart_type='ohlc'):
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ] 
    # create a plot
    fig, ax = plt.subplots(figsize = (10, 5))  
    # thin black bars for better long-term visualization
    if plot_type == 'bars':
        for i in sample.index:  
            plt.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                plt.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                plt.vlines(x=i, ymin = sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                plt.vlines(x=i, ymin = sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)  
    # regular candlesticks for better interpretation
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            plt.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                plt.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                plt.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                plt.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+0.5, color='black', linewidth=3)  
    # simple line chart using the open prices (to choose close, switch the below argument)
    elif plot_type == 'line':
        if chart_type == 'ohlc':
            plt.plot(sample['open'], color='black')
        elif chart_type == 'simple_economic_indicator':
            plt.plot(sample['value'], color='black')
        elif chart_type == 'simple_financial':
            plt.plot(sample['close'], color='black')
    else:
        print('Choose between bars or candlesticks')           
    plt.grid()
    plt.show()
    plt.tight_layout()

def signal_chart(my_time_series, window, choice='bars', source='open', chart_type='ohlc'): 
    # choose a sampling window
    sample = my_time_series.iloc[-window:, ]
    if chart_type == 'ohlc':
        ohlc_plot(sample, window, plot_type=choice)    
        for i in my_time_series.index:
            if my_time_series.loc[i, 'bullish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]-1),
                            arrowprops=dict(facecolor='green', shrink=0.05))
            elif my_time_series.loc[i, 'bearish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]+1),
                            arrowprops=dict(facecolor='red', shrink=0.05))    
    elif chart_type == 'simple':
        ohlc_plot(sample, window, plot_type = 'line', chart_type='simple')  
        for i in my_time_series.index:
            if my_time_series.loc[i, 'bullish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]-1),
                            arrowprops=dict(facecolor='green', shrink=0.05))
            elif my_time_series.loc[i, 'bearish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]+1),
                            arrowprops=dict(facecolor='red', shrink=0.05)) 
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    plt.legend(handles=[bullish_signal, bearish_signal])
    plt.tight_layout()

def signal_chart_retracement(my_time_series, window=250): 
    sample = my_time_series.iloc[-window:, ]
    ohlc_plot(sample, window, plot_type='bars')      
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='o', s=50, zorder=3, label = 'Swing Point')
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='o', s=50, zorder=3)
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']+1),
                        arrowprops=dict(facecolor='red', shrink=0.05)) 
    for i, row in sample.iterrows():
        if not np.isnan(row['support']):
            x = i
            y = row['support']
            plt.hlines(y, xmin=x, xmax=x + pd.Timedelta(days=5), color='darkgrey', linewidth=2)
    for i, row in sample.iterrows():
        if not np.isnan(row['resistance']):
            x = i
            y = row['resistance']
            plt.hlines(y, xmin=x, xmax=x + pd.Timedelta(days=5), color='black', linewidth=2)
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    swing_point = Line2D([0], [0], marker='o', color='w', label='Swing Points', markerfacecolor='blue', markersize=10)    
    plt.legend(handles=[bullish_signal, bearish_signal, swing_point])
    plt.tight_layout()
            
def indicator_plot(my_time_series, indicator, window=500, lower_barrier=30, upper_barrier=70, plot_type='bars', barriers=True,
                     indicator_label='Indicator'): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]   
    if plot_type == 'bars':
        for i in sample.index:  
            ax1.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin = sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)   
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            ax1.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=5)  
    elif plot_type == 'line':
        ax1.plot(sample['open'], color='black') 
    else:
        print('Choose between bars or candlesticks')           
    ax2.plot(sample.index, sample[indicator], label=indicator_label, color='blue')
    if barriers == True:
        ax2.axhline(y=lower_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=upper_barrier, color='black', linestyle='dashed')
    ax1.grid()
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    
def signal_chart_indicator(my_time_series, indicator, window=500, lower_barrier=30, upper_barrier=70, plot_type='bars',
                             barriers=True, indicator_label='Indicator'): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]   
    if plot_type == 'bars':
        for i in sample.index:  
            ax1.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin = sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)   
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            ax1.vlines(x=i, ymin =sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=5)  
    elif plot_type == 'line':
        ax1.plot(sample['open'], color='black') 
    else:
        print('Choose between bars or candlesticks')           
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))    
    ax2.plot(sample.index, sample[indicator], label=indicator_label, color='blue')
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    ax1.legend(handles=[bullish_signal, bearish_signal])
    if barriers == True:
        ax2.axhline(y=lower_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=upper_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=50, color='black', linestyle='dashed')        
    plt.tight_layout()
    ax1.grid()
    ax2.legend()
    ax2.grid()
    
def moving_average(my_time_series, source='close', ma_lookback=200, output_name='moving_average', ma_type='SMA'):
    '''
    The moving averages available in this function:
        * Simple moving average (SMA)
        * Exponential moving average (EMA)
        * Smoothed moving average (SMMA)
    '''
    if source not in my_time_series.columns:
        raise ValueError(f"Column '{source}' not found in DataFrame, choose SMA, EMA, or SMMA")
    # calculate the simple moving average
    if ma_type == 'SMA':
        my_time_series[output_name] = my_time_series[source].rolling(window=ma_lookback).mean()
        return my_time_series.dropna()
    # calculate the exponential moving average
    elif ma_type == 'EMA':
        my_time_series[output_name] = my_time_series[source].ewm(span=ma_lookback, adjust=False).mean()
        return my_time_series.dropna()
    # calculate the smoothed moving average
    elif ma_type == 'SMMA':
        my_time_series[output_name] = my_time_series[source].ewm(span=(ma_lookback*2)-1, adjust=False).mean()
        return my_time_series.dropna()
    else:
        raise ValueError("ma_type must be either 'SMA', 'EMA', or 'SMMA'")
        
def rsi(my_time_series, source='close', output_name='RSI', rsi_lookback=14):
    # calculating the difference between the close prices at each time step
    delta = my_time_series[source].diff(1)
    # isolating the positive differences and the absolute negative differences
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # transforming the exponential moving average to a smoothed moving average
    rsi_real_lookback = (rsi_lookback*2)-1
    # calculating a rolling smoothed moving average on the gain and loss variables
    avg_gain = gain.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_real_lookback, min_periods=1, adjust=False).mean()
    # calculating the relative strength
    rs = avg_gain / avg_loss 
    # calculating the relative strength index
    rsi = 100-(100/(1+rs))
    # creating a column in the data frame and populating it with the RSI
    my_time_series[output_name] = rsi
    return my_time_series.dropna()

def rsi_square(my_time_series, source='close', rsi_prime_lookback=14, rsi_square_lookback=5):
    my_time_series = rsi(my_time_series, source=source, output_name='RSI', rsi_lookback=rsi_prime_lookback)
    my_time_series = rsi(my_time_series, source='RSI', output_name='RSI²', rsi_lookback=rsi_prime_lookback)
    return my_time_series.dropna()

def atr(my_time_series, vol_lookback=20):
    # create a column containing the previous close prices
    my_time_series['previous_close'] = my_time_series['close'].shift(1)
    # calculate the true range
    my_time_series['TR'] = my_time_series.apply(lambda row: max(row['high'] - row['low'],
                                                            abs(row['high'] - row['previous_close']),
                                                            abs(row['low'] - row['previous_close'])), axis=1)
    # transform the lookback to fit a smoothed moving average
    vol_lookback = (vol_lookback * 2) - 1
    # calculate the atr
    my_time_series['volatility'] = my_time_series['TR'].ewm(span=vol_lookback, adjust=False).mean()
    my_time_series = my_time_series.drop(columns=['previous_close', 'TR'])    
    return my_time_series.dropna()

def atr_adjusted_rsi(my_time_series, source='close', rsi_lookback=13, vol_lookback=5, rsi_atr_lookback=13):
    my_time_series = rsi(my_time_series, source='close', output_name='RSI', rsi_lookback=rsi_lookback)
    my_time_series = atr(my_time_series, vol_lookback=vol_lookback)
    my_time_series['RSI_times_atr'] = my_time_series['RSI'] * my_time_series['volatility']
    my_time_series = rsi(my_time_series, source='RSI_times_atr', output_name='atr_adjusted_rsi', rsi_lookback=rsi_atr_lookback)
    return my_time_series.dropna()

def heikin_ashi(my_time_series):
    # close price using heikin-ashi
    my_time_series['HA_close'] = (my_time_series['open'] + my_time_series['high'] + my_time_series['low'] + my_time_series['close']) / 4
    # open price using heikin-ashi
    my_time_series['HA_open'] = 0
    my_time_series['HA_open'].iloc[0] = my_time_series['open'].iloc[0]    
    for i in range(1, len(my_time_series)):
        my_time_series.at[my_time_series.index[i], 'HA_open'] = (my_time_series['HA_open'].iloc[i-1] + \
                                                                 my_time_series['HA_close'].iloc[i-1]) / 2
    # high price using heikin-ashi
    my_time_series['HA_high'] = my_time_series[['high', 'HA_open', 'HA_close']].max(axis=1)
    # low price using heikin-ashi
    my_time_series['HA_low'] = my_time_series[['low', 'HA_open', 'HA_close']].min(axis=1)
    return my_time_series.dropna()
    
def k_candlesticks(my_time_series, k_lookback=5):
    # calculating the exponential moving average on the ohlc data
    my_time_series = moving_average(my_time_series, 'open',  ma_lookback=k_lookback, output_name='k_open',  ma_type='EMA')
    my_time_series = moving_average(my_time_series, 'high',  ma_lookback=k_lookback, output_name='k_high',  ma_type='EMA') 
    my_time_series = moving_average(my_time_series, 'low',   ma_lookback=k_lookback, output_name='k_low',   ma_type='EMA')
    my_time_series = moving_average(my_time_series, 'close', ma_lookback=k_lookback, output_name='k_close', ma_type='EMA')
    return my_time_series.dropna()

def td_setup(my_time_series, source='close', perfected_source_low='low', perfected_source_high='high', final_step=9, 
              difference=4, perfected=False):
    my_time_series['buy_setup'] = 0
    my_time_series['sell_setup'] = 0
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    # show perfected and imperfected setups
    if perfected == False:
        for i in range(4, len(my_time_series)):
            # bullish setup
            if my_time_series[source].iloc[i] < my_time_series[source].iloc[i-difference]:
                my_time_series.at[my_time_series.index[i], 'buy_setup'] = my_time_series['buy_setup'].iloc[i-1] + 1 if my_time_series['buy_setup'].iloc[i-1] < final_step else 0
            else:
                my_time_series.at[my_time_series.index[i], 'buy_setup'] = 0
            # bearish setup
            if my_time_series[source].iloc[i] > my_time_series[source].iloc[i-difference]:
                my_time_series.at[my_time_series.index[i], 'sell_setup'] = my_time_series['sell_setup'].iloc[i-1] + 1 if my_time_series['sell_setup'].iloc[i-1] < final_step else 0
            else:
                my_time_series.at[my_time_series.index[i], 'sell_setup'] = 0   
        for i in range(4, len(my_time_series)):
            try:
                if my_time_series['buy_setup'].iloc[i] == final_step:
                    my_time_series['bullish_signal'].iloc[i+1] = 1
                elif my_time_series['sell_setup'].iloc[i] == final_step:
                    my_time_series['bearish_signal'].iloc[i+1] = 1
            except (KeyError, IndexError):
                pass
    # show only perfected setups        
    if perfected == True:
        for i in range(4, len(my_time_series)):
            # bullish setup
            if my_time_series[source].iloc[i] < my_time_series[source].iloc[i-difference]:
                my_time_series.at[my_time_series.index[i], 'buy_setup'] = my_time_series['buy_setup'].iloc[i-1] + 1 if my_time_series['buy_setup'].iloc[i-1] < final_step else 0
            else:
                my_time_series.at[my_time_series.index[i], 'buy_setup'] = 0
            # bearish setup
            if my_time_series[source].iloc[i] > my_time_series[source].iloc[i - difference]:
                my_time_series.at[my_time_series.index[i], 'sell_setup'] = my_time_series['sell_setup'].iloc[i-1] + 1 if my_time_series['sell_setup'].iloc[i-1] < final_step else 0
            else:
                my_time_series.at[my_time_series.index[i], 'sell_setup'] = 0   
        for i in range(4, len(my_time_series)):
            try:
                if my_time_series['buy_setup'].iloc[i] == final_step and \
                   my_time_series[perfected_source_low].iloc[i] < my_time_series[perfected_source_low].iloc[i-2] and \
                   my_time_series[perfected_source_low].iloc[i] < my_time_series[perfected_source_low].iloc[i-3]:
                    my_time_series['bullish_signal'].iloc[i+1] = 1
                elif my_time_series['sell_setup'].iloc[i] == final_step and \
                   my_time_series[perfected_source_high].iloc[i] > my_time_series[perfected_source_high].iloc[i-2] and \
                   my_time_series[perfected_source_high].iloc[i] > my_time_series[perfected_source_high].iloc[i-3]:
                    my_time_series['bearish_signal'].iloc[i+1] = 1
            except (KeyError, IndexError):
                pass     
    return my_time_series

def fibonacci_timing_pattern(my_time_series, source='close', final_step=8, first_difference=5, second_difference=21):
    my_time_series['buy_setup'] = 0
    my_time_series['sell_setup'] = 0
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish setup
        if my_time_series[source].iloc[i] < my_time_series[source].iloc[i - first_difference] and \
           my_time_series[source].iloc[i - first_difference] < my_time_series[source].iloc[i - second_difference]:
            my_time_series.at[my_time_series.index[i], 'buy_setup'] = my_time_series['buy_setup'].iloc[i - 1] + 1 if my_time_series['buy_setup'].iloc[i - 1] < final_step else 0
        else:
            my_time_series.at[my_time_series.index[i], 'buy_setup'] = 0
        # bearish setup
        if my_time_series[source].iloc[i] > my_time_series[source].iloc[i - first_difference] and \
           my_time_series[source].iloc[i - first_difference] > my_time_series[source].iloc[i - second_difference]:
            my_time_series.at[my_time_series.index[i], 'sell_setup'] = my_time_series['sell_setup'].iloc[i - 1] + 1 if my_time_series['sell_setup'].iloc[i - 1] < final_step else 0
        else:
            my_time_series.at[my_time_series.index[i], 'sell_setup'] = 0   
    for i in range(0, len(my_time_series)):
        try:
            if my_time_series['buy_setup'].iloc[i] == final_step:
                my_time_series['bullish_signal'].iloc[i+1] = 1
            elif my_time_series['sell_setup'].iloc[i] == final_step:
                my_time_series['bearish_signal'].iloc[i+1] = 1
        except (KeyError, IndexError):
            pass
    return my_time_series

def macd(my_time_series, source='close', short_window=12, long_window=26, signal_window=9):
    # calculate the short-term EMA
    my_time_series['EMA_short'] = my_time_series[source].ewm(span=short_window, adjust=False).mean()
    # calculate the long-term EMA 
    my_time_series['EMA_long'] = my_time_series[source].ewm(span=long_window, adjust=False).mean()
    # calculate the MACD line
    my_time_series['MACD_line'] = my_time_series['EMA_short'] - my_time_series['EMA_long']
    # calculate the Signal line
    my_time_series['MACD_signal'] = my_time_series['MACD_line'].ewm(span=signal_window, adjust=False).mean()
    # calculate the MACD Histogram
    my_time_series['MACD_histogram'] = my_time_series['MACD_line'] - my_time_series['MACD_signal']
    # drop the EMA columns as they are not needed anymore
    my_time_series.drop(['EMA_short', 'EMA_long'], axis=1, inplace=True)
    return my_time_series.dropna()

def bollinger_bands(my_time_series, source='close', bb_lookback=20, num_std_dev=2):
    # calculate the moving average
    my_time_series['middle_band'] = my_time_series[source].rolling(window=bb_lookback).mean()
    # calculate the rolling standard deviation
    my_time_series['volatility'] = my_time_series[source].rolling(window=bb_lookback).std()
    # calculate the upper bollinger band
    my_time_series['upper_band'] = my_time_series['middle_band'] + (my_time_series['volatility'] * num_std_dev)
    # calculate the lower bollinger band
    my_time_series['lower_band'] = my_time_series['middle_band'] - (my_time_series['volatility'] * num_std_dev)
    # drop the rolling standard deviation column as it's not typically needed
    my_time_series.drop(['volatility'], axis=1, inplace=True)
    return my_time_series.dropna()

def candlestick_rsi(my_time_series, rsi_lookback=14):
    my_time_series = rsi(my_time_series, source='open', output_name='transformed_open', rsi_lookback=rsi_lookback)
    my_time_series = rsi(my_time_series, source='high', output_name='transformed_high', rsi_lookback=rsi_lookback)
    my_time_series = rsi(my_time_series, source='low', output_name='transformed_low', rsi_lookback=rsi_lookback)
    my_time_series = rsi(my_time_series, source='close', output_name='transformed_close', rsi_lookback=rsi_lookback)
    # rsi - high
    my_time_series['RSI_high'] = my_time_series[['transformed_open', 
                                                 'transformed_high',
                                                 'transformed_low',
                                                 'transformed_close']].max(axis=1)    
    # rsi - low
    my_time_series['RSI_low'] = my_time_series[['transformed_open', 
                                                 'transformed_high',
                                                 'transformed_low',
                                                 'transformed_close']].min(axis=1)
    # rsi - close/open
    for i in range(0, len(my_time_series)):
        if my_time_series['close'].iloc[i] > my_time_series['open'].iloc[i]:
            my_time_series.at[my_time_series.index[i], 'RSI_close'] = max(my_time_series['transformed_open'].iloc[i], \
                                                                          my_time_series['transformed_close'].iloc[i])
            my_time_series.at[my_time_series.index[i], 'RSI_open'] = min(my_time_series['transformed_open'].iloc[i], \
                                                                         my_time_series['transformed_close'].iloc[i])                
        elif my_time_series['close'].iloc[i] < my_time_series['open'].iloc[i]:
            my_time_series.at[my_time_series.index[i], 'RSI_close'] = min(my_time_series['transformed_open'].iloc[i], \
                                                                          my_time_series['transformed_close'].iloc[i]) 
            my_time_series.at[my_time_series.index[i], 'RSI_open'] = max(my_time_series['transformed_open'].iloc[i], \
                                                                         my_time_series['transformed_close'].iloc[i])                
    my_time_series.drop(['transformed_open', 'transformed_high', 'transformed_low', 'transformed_close'], axis=1, inplace=True)
    return my_time_series.dropna()

def signal_chart_candlestick_rsi(my_time_series, window=500, lower_barrier=30, upper_barrier=70, plot_type='bars', barriers=True): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]   
    if plot_type == 'bars':
        for i in sample.index:  
            ax1.vlines(x = i, ymin = sample.at[i, 'low'], ymax = sample.at[i, 'high'], color = 'black', linewidth = 1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'black', linewidth = 1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'black', linewidth =1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'] + 1, color = 'black', linewidth = 1)   
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            ax1.vlines(x = i, ymin = sample.at[i, 'low'], ymax = sample.at[i, 'high'], color = 'black', linewidth = 1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'green', linewidth = 3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'red', linewidth = 3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'] + 3, color = 'black', linewidth = 5)  
    elif plot_type == 'line':
        ax1.plot(sample['open'], color = 'black') 
    else:
        print('Choose between bars or candlesticks')           
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))
    for i in sample.index:  
        ax2.vlines(x = i, ymin = sample.at[i, 'RSI_low'], ymax = sample.at[i, 'RSI_high'], color = 'black', linewidth = 1)  
        if sample.at[i, 'RSI_close'] > sample.at[i, 'RSI_open']: 
            ax2.vlines(x = i, ymin = sample.at[i, 'RSI_open'], ymax = sample.at[i, 'RSI_close'], color = 'green', linewidth = 3)  
        if sample.at[i, 'RSI_close'] < sample.at[i, 'RSI_open']:
            ax2.vlines(x = i, ymin = sample.at[i, 'RSI_close'], ymax = sample.at[i, 'RSI_open'], color = 'red', linewidth = 3)   
        if sample.at[i, 'RSI_close'] == sample.at[i, 'RSI_open']:
            ax2.vlines(x = i, ymin = sample.at[i, 'RSI_close'], ymax = sample.at[i, 'RSI_open'], color = 'black', linewidth = 2)
    if barriers == True:
        ax2.axhline(y = lower_barrier, color = 'black', linestyle = 'dashed')
        ax2.axhline(y = upper_barrier, color = 'black', linestyle = 'dashed')
        ax2.axhline(y = 50, color = 'black', linestyle = 'dashed')        
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    
def stochastic_oscillator(my_time_series, k_lookback=14, k_smoothing_lookback=3, d_lookback=3):
    my_time_series['lowest_low']   = my_time_series['low'].rolling(window=k_lookback).min()
    my_time_series['highest_high'] = my_time_series['high'].rolling(window=k_lookback).max()   
    my_time_series['%K'] = 100 * ((my_time_series['close'] - my_time_series['lowest_low']) / \
                                  (my_time_series['highest_high'] - my_time_series['lowest_low']))
    my_time_series['%K_smoothing'] = my_time_series['%K'].rolling(window=k_smoothing_lookback).mean()
    my_time_series['%D'] = my_time_series['%K_smoothing'].rolling(window=d_lookback).mean()
    my_time_series = my_time_series.drop(columns=['lowest_low', 'highest_high'])
    return my_time_series.dropna()

def import_data_with_volume(name='NVDA', start_date='2020-01-01', end_date='2025-07-01'):
    import yfinance as yf
    my_time_series = yf.download(name, start=start_date, end=end_date)
    my_time_series = my_time_series.xs(name, level=1, axis=1)
    my_time_series.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'}, inplace=True)
    return my_time_series

def normalize(my_time_series, source='close', output_column='normalized', normalized_lookback=20):
    rolling_min = my_time_series[source].rolling(window=normalized_lookback).min()
    rolling_max = my_time_series[source].rolling(window=normalized_lookback).max()
    my_time_series[output_column] = (my_time_series[source] - rolling_min) / (rolling_max - rolling_min)
    return my_time_series.dropna()

def abcd_pattern(my_time_series, swing_lookback=20):
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    # initialize tracking variables
    prev1 = None
    prev2 = None
    prev3 = None
    # labeling
    for i, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swing_type = 'high'
            swing_value = row['swing_high']
        elif not pd.isna(row['swing_low']):
            swing_type = 'low'
            swing_value = row['swing_low']
        else:
            continue
        # skip if same type as previous swing
        if prev1 and swing_type == prev1[1]:
            continue
        # shift swings
        prev3 = prev2
        prev2 = prev1
        prev1 = (i, swing_type, swing_value)
        # check if we have 3 alternating swings to form ABC
        if prev3:
            A_i, A_type, A_val = prev3
            B_i, B_type, B_val = prev2
            C_i, C_type, C_val = prev1
            # bullish ABCD pattern: high → low → high → projecting low
            if A_type == 'high' and B_type == 'low' and C_type == 'high':
                if C_val <= A_val:
                    projection = C_val - (A_val - B_val)
                    my_time_series.at[C_i, 'bullish_signal'] = projection
            # bearish ABCD pattern: low → high → low → projecting high
            elif A_type == 'low' and B_type == 'high' and C_type == 'low':
                if C_val >= A_val:                
                    projection = C_val + (B_val - A_val)
                    my_time_series.at[C_i, 'bearish_signal'] = projection
    return my_time_series

def signal_chart_abcd(my_time_series, window=250, choice='bars', source='open'): 
    sample = my_time_series.iloc[-window:, ]
    sample['zigzag'] = sample['swing_high'].combine_first(sample['swing_low'])
    zigzag = sample['zigzag'].dropna()
    ohlc_plot(sample, window, plot_type=choice)   
    plt.plot(zigzag.index, zigzag, color='orange', linewidth=2, label='ZigZag', linestyle='dotted')     
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='.', s=50, zorder=3)
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='.', s=50, zorder=3)
    for i, row in sample.iterrows():
        if not np.isnan(row['bullish_signal']):
            x = i
            y = row['bullish_signal']
            plt.hlines(y, xmin=x+pd.Timedelta(days=5), xmax=x + pd.Timedelta(days=50), color='darkgrey', linewidth=2)
            plt.plot(x+pd.Timedelta(days=25), y, marker='^', color='darkgrey', markersize=10)
    for i, row in sample.iterrows():
        if not np.isnan(row['bearish_signal']):
            x = i
            y = row['bearish_signal']
            plt.hlines(y, xmin=x+pd.Timedelta(days=5), xmax=x + pd.Timedelta(days=50), color='black', linewidth=2)    
            plt.plot(x+pd.Timedelta(days=25), y, marker='v', color='black', markersize=10)
    from matplotlib.lines import Line2D
    support = Line2D([0], [0], marker='_', color='darkgrey', label='Support', markerfacecolor='darkgrey', markersize=10)
    resistance = Line2D([0], [0], marker='_', color='black', label='Resistance', markerfacecolor='black', markersize=10)
    swing_point = Line2D([0], [0], marker='.', color='w', label='Swing Points', markerfacecolor='blue', markersize=10) 
    zig_zag = Line2D([0], [0], marker='_', color='orange', label='Zig Zag', markerfacecolor='orange', markersize=10)     
    plt.legend(handles=[support, resistance, swing_point, zig_zag])
    plt.tight_layout()

def signal_chart_abcd_trend(my_time_series, window=250, source='open', choice='candlesticks'): 
    sample = my_time_series.iloc[-window:, ]
    sample['zigzag'] = sample['swing_high'].combine_first(sample['swing_low'])
    zigzag = sample['zigzag'].dropna()
    ohlc_plot(sample, window, plot_type=choice)   
    plt.plot(zigzag.index, zigzag, color='orange', linewidth=2, label='ZigZag', linestyle='dotted')  
    plt.plot(sample.index, sample['moving_average'], color='purple', linewidth=2, label='Moving Average', linestyle='dashed')     
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='.', s=50, zorder=3)
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='.', s=50, zorder=3)
    for i, row in sample.iterrows():
        if not np.isnan(row['abcd_bullish_projection_trend']):
            x = i
            y = row['abcd_bullish_projection_trend']
            plt.hlines(y, xmin=x+pd.Timedelta(days=5), xmax=x + pd.Timedelta(days=15), color='darkgrey', linewidth=2)
            plt.plot(x+pd.Timedelta(days=10), y, marker='^', color='darkgrey', markersize=10)
    for i, row in sample.iterrows():
        if not np.isnan(row['abcd_bearish_projection_trend']):
            x = i
            y = row['abcd_bearish_projection_trend']
            plt.hlines(y, xmin=x+pd.Timedelta(days=5), xmax=x + pd.Timedelta(days=15), color='black', linewidth=2)    
            plt.plot(x+pd.Timedelta(days=10), y, marker='v', color='black', markersize=10)
    from matplotlib.lines import Line2D
    support = Line2D([0], [0], marker='_', color='darkgrey', label='Support', markerfacecolor='darkgrey', markersize=10)
    resistance = Line2D([0], [0], marker='_', color='black', label='Resistance', markerfacecolor='black', markersize=10)
    swing_point = Line2D([0], [0], marker='.', color='w', label='Swing Points', markerfacecolor='blue', markersize=10) 
    zig_zag = Line2D([0], [0], marker='_', color='orange', label='Zig Zag', markerfacecolor='orange', markersize=10)  
    ma = Line2D([0], [0], marker='_', color='purple', label='Moving Average', markerfacecolor='purple', markersize=10)      
    plt.legend(handles=[support, resistance, swing_point, zig_zag, ma])
    plt.tight_layout()
    
def generate_synthetic_symmetric_data(num_data=200, amplitude=5, base_price=100):
    frequency = 2 * np.pi / 20
    index = pd.date_range(start='2024-01-01', periods=num_data, freq='D')
    sin_wave = amplitude * np.sin(frequency * np.arange(num_data))
    price = base_price + sin_wave
    open_price = price
    close_price = open_price + np.random.normal(0, 0.5, size=num_data)
    high_price = np.maximum(open_price, close_price) + np.random.uniform(0.2, 1.0, size=num_data)
    low_price = np.minimum(open_price, close_price) - np.random.uniform(0.2, 1.0, size=num_data)
    my_time_series = pd.DataFrame({'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price}, index=index)
    my_time_series = my_time_series.round(2)
    return my_time_series
    
def slope(my_time_series, source='close', output_name='slope', slope_lookback=14):
    my_time_series[output_name] = ''
    for i in range(len(my_time_series)): 
        my_time_series[output_name].iloc[i] = (my_time_series[source].iloc[i] - my_time_series[source].iloc[i-slope_lookback])/slope_lookback
    return my_time_series.dropna()

def swing_detect(my_time_series, swing_lookback=20):
    my_time_series['swing_low']  = my_time_series['low'].rolling(window=swing_lookback, min_periods=1, center=True).min()
    my_time_series['swing_low']  = my_time_series.apply(lambda row: row['low'] if row['low'] == row['swing_low'] else 0, axis=1)
    my_time_series['swing_low']  = my_time_series['swing_low'].replace(0, np.nan)
    my_time_series['swing_high'] = my_time_series['high'].rolling(window=swing_lookback, min_periods=1, center=True).max()
    my_time_series['swing_high'] = my_time_series.apply(lambda row: row['high'] if row['high'] == row['swing_high'] else 0, axis=1)
    my_time_series['swing_high'] = my_time_series['swing_high'].replace(0, np.nan)
    return my_time_series

def zig_zag(my_time_series):
    my_time_series = swing_detect(my_time_series, lookback=20)
    my_time_series = my_time_series.iloc[-250:, ] 
    my_time_series['zigzag'] = my_time_series['swing_high'].combine_first(my_time_series['swing_low'])
    zigzag = my_time_series['zigzag'].dropna()
    ohlc_plot(my_time_series, window=250, plot_type='bars', chart_type='ohlc')
    plt.plot(zigzag.index, zigzag, color='red', linewidth=2, label='ZigZag')
    plt.scatter(my_time_series.index, my_time_series['swing_high'], marker='o', color='blue', label='Swing Highs/Lows')
    plt.scatter(my_time_series.index, my_time_series['swing_low'], marker='o', color='blue') 
    plt.legend()
    plt.tight_layout()
    plt.show()

def swing_points_chart(my_time_series, window=250):
    sample = my_time_series.iloc[-window:, ]
    ohlc_plot(sample, plot_type='bars', chart_type='ohlc')
    plt.scatter(sample.index, sample['swing_high'], marker='o', color='blue', label='Swing Highs/Lows')
    plt.scatter(sample.index, sample['swing_low'], marker='o', color='blue') 
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def fibonacci_retracement(my_time_series, swing_lookback=20, fib_level=0.236):
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    my_time_series['support'] = np.nan
    my_time_series['resistance'] = np.nan    
    last_swing_type = None
    last_swing_value = None
    for i, row in my_time_series.iterrows():
        if not np.isnan(row['swing_high']):
            current_type = 'high'
            current_value = row['swing_high']
        elif not np.isnan(row['swing_low']):
            current_type = 'low'
            current_value = row['swing_low']
        else:
            continue    
        if last_swing_type and current_type != last_swing_type:
            if current_type == 'low':
                # last was high → current is low
                my_time_series.at[i, 'resistance'] = ((last_swing_value - current_value) * fib_level) + current_value
            else:
                # last was low → current is high
                my_time_series.at[i, 'support'] = current_value - ((current_value - last_swing_value) * fib_level) 
        last_swing_type = current_type
        last_swing_value = current_value
    return my_time_series

def display_retracement(my_time_series, window=250, choice='bars'): 
    sample = my_time_series.iloc[-window:, ]
    ohlc_plot(sample, window, plot_type=choice)      
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='o', s=50, zorder=3)
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='o', s=50, zorder=3)
    for i, row in sample.iterrows():
        if not np.isnan(row['support']):
            x = i
            y = row['support']
            plt.hlines(y, xmin=x, xmax=x + pd.Timedelta(days=15), color='darkgrey', linewidth=2)
    for i, row in sample.iterrows():
        if not np.isnan(row['resistance']):
            x = i
            y = row['resistance']
            plt.hlines(y, xmin=x, xmax=x + pd.Timedelta(days=15), color='black', linewidth=2)
    from matplotlib.lines import Line2D
    support = Line2D([0], [0], marker='_', color='darkgrey', label='Support', markerfacecolor='darkgrey', markersize=10)
    resistance = Line2D([0], [0], marker='_', color='black', label='Resistance', markerfacecolor='black', markersize=10)
    swing_point = Line2D([0], [0], marker='o', color='w', label='Swing Points', markerfacecolor='blue', markersize=10) 
    plt.legend(handles=[support, resistance, swing_point])
    plt.tight_layout()
    
def retracement_signals(my_time_series, technique='reintegration'): 
    if technique=='reintegration':
        my_time_series['bullish_signal'] = 0
        my_time_series['bearish_signal'] = 0
        for i in range(0, len(my_time_series)):
            # bullish signal
            if my_time_series['close'].iloc[i] > my_time_series['resistance'].iloc[i-1]:
                my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
            # bearish signal
            elif my_time_series['close'].iloc[i] < my_time_series['support'].iloc[i]:
                my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    elif technique=='reactionary':
        my_time_series['bullish_signal'] = 0
        my_time_series['bearish_signal'] = 0
        for i in range(0, len(my_time_series)):
            # bullish signal
            if my_time_series['close'].iloc[i] > my_time_series['support'].iloc[i-1] and \
               my_time_series['close'].iloc[i-1] < my_time_series['support'].iloc[i-1]:
                my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
            # bearish signal
            elif my_time_series['close'].iloc[i] < my_time_series['resistance'].iloc[i] and \
                 my_time_series['close'].iloc[i-1] < my_time_series['resistance'].iloc[i-1]:
                my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1        
    return my_time_series
            
def standard_deviation(my_time_series, source='close', vol_lookback=20):
    my_time_series['volatility'] = my_time_series[source].rolling(window=vol_lookback).std()
    return my_time_series.dropna() 

def exponentially_weighted_standard_deviation(my_time_series, source='close', vol_lookback=20):
    my_time_series['volatility'] = my_time_series[source].ewm(span=vol_lookback, adjust=False).std()
    return my_time_series.dropna()

def spike_weighted_volatility(my_time_series, source='close', vol_lookback=20):
    returns = my_time_series[source].pct_change()
    mean_returns = returns.rolling(vol_lookback).mean()
    std_returns = returns.rolling(vol_lookback).std()
    spike_factor = np.abs(returns - mean_returns) / (std_returns + 1e-8)
    weighted_squared = (returns ** 2) * (1 + spike_factor)
    aswv = np.sqrt(weighted_squared.rolling(vol_lookback).mean())
    my_time_series['swv'] = aswv
    return my_time_series.dropna()

def hma(my_time_series, ma_lookback=50):
    # half period WMA as integer
    half_lookback = ma_lookback // 2
    # calculate WMA for half period
    my_time_series['WMA_half'] = wma(my_time_series, ma_lookback=half_lookback)
    # calculate WMA for full period
    my_time_series['WMA_full'] = wma(my_time_series, ma_lookback=ma_lookback)
    # HMA calculation
    my_time_series['HMA_numerator'] = 2 * my_time_series['WMA_half'] - my_time_series['WMA_full']
    hma_lookback = int(np.sqrt(ma_lookback))
    my_time_series['HMA'] = wma(my_time_series, source='HMA_numerator', ma_lookback=hma_lookback)
    # drop the initial columns as they are not needed anymore
    my_time_series.drop(['WMA_half', 'WMA_full', 'HMA_numerator'], axis=1, inplace=True)
    return my_time_series.dropna()

def iwma(my_time_series, source='close', ma_lookback=50):
    # generate weights based on the lookback period
    weights = np.arange(1, ma_lookback + 1)[::-1]   
    # create an empty series to store IWMA values
    my_time_series['IWMA'] = 0  
    # compute IWMA for each rolling window
    for i in range(ma_lookback - 1, len(my_time_series)):
        window = my_time_series[source].iloc[i - ma_lookback + 1:i + 1]
        weighted_sum = np.dot(window, weights)
        sum_of_weights = weights.sum()
        my_time_series['IWMA'].iloc[i] = weighted_sum / sum_of_weights
    return my_time_series['IWMA'].dropna()

def wma(my_time_series, source='close', ma_lookback=50):
    # generate weights based on the lookback period
    weights = np.arange(1, ma_lookback + 1)   
    # create an empty series to store WMA values
    my_time_series['WMA'] = 0
    # compute WMA for each rolling window
    for i in range(ma_lookback - 1, len(my_time_series)):
        window = my_time_series[source].iloc[i - ma_lookback + 1:i + 1]
        weighted_sum = np.dot(window, weights)
        sum_of_weights = weights.sum()
        my_time_series['WMA'].iloc[i] = weighted_sum / sum_of_weights
    return my_time_series['WMA'].dropna()

def e_bollinger_bands(my_time_series, source='close', bb_lookback=20, num_std_dev=2):
    # calculate the exponential moving average
    my_time_series['middle_band'] = my_time_series[source].ewm(span=bb_lookback, adjust=False).mean()
    # calculate the rolling standard deviation
    my_time_series['volatility'] = my_time_series[source].rolling(window=bb_lookback).std()
    # calculate the upper bollinger band
    my_time_series['upper_band'] = my_time_series['middle_band'] + (my_time_series['volatility'] * num_std_dev)
    # calculate the lower bollinger band
    my_time_series['lower_band'] = my_time_series['middle_band'] - (my_time_series['volatility'] * num_std_dev)
    # drop the rolling standard deviation column as it's not typically needed
    my_time_series.drop(['volatility'], axis=1, inplace=True)
    return my_time_series.dropna()

def red_indicator(my_time_series):
    my_time_series = e_bollinger_bands(my_time_series)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['close'].iloc[i] < my_time_series['middle_band'].iloc[i] and \
           my_time_series['close'].iloc[i] > my_time_series['lower_band'].iloc[i] and \
           my_time_series['close'].iloc[i-1] < my_time_series['lower_band'].iloc[i-1] and \
           my_time_series['close'].iloc[i-2] < my_time_series['lower_band'].iloc[i-2] and \
           my_time_series['close'].iloc[i-3] < my_time_series['lower_band'].iloc[i-3]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['close'].iloc[i] > my_time_series['middle_band'].iloc[i] and \
             my_time_series['close'].iloc[i] < my_time_series['upper_band'].iloc[i] and \
             my_time_series['close'].iloc[i-1] > my_time_series['upper_band'].iloc[i-1] and \
             my_time_series['close'].iloc[i-2] > my_time_series['upper_band'].iloc[i-2] and \
             my_time_series['close'].iloc[i-3] > my_time_series['upper_band'].iloc[i-3]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def orange_indicator(my_time_series, column='close', output_name='RSI', rsi_lookback=8, lower_barrier=35, upper_barrier=65):
    my_time_series = rsi(my_time_series, column, output_name, rsi_lookback=rsi_lookback)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['RSI'].iloc[i] > lower_barrier and \
           my_time_series['RSI'].iloc[i] < 50 and \
           my_time_series['RSI'].iloc[i-1] < lower_barrier and \
           my_time_series['RSI'].iloc[i-2] < lower_barrier and \
           my_time_series['RSI'].iloc[i-3] < lower_barrier and \
           my_time_series['RSI'].iloc[i-4] < lower_barrier and \
           my_time_series['RSI'].iloc[i-5] < lower_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['RSI'].iloc[i] < upper_barrier and \
             my_time_series['RSI'].iloc[i] > 50 and \
             my_time_series['RSI'].iloc[i-1] > upper_barrier and \
             my_time_series['RSI'].iloc[i-2] > upper_barrier and \
             my_time_series['RSI'].iloc[i-3] > upper_barrier and \
             my_time_series['RSI'].iloc[i-4] > upper_barrier and \
             my_time_series['RSI'].iloc[i-5] > upper_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def yellow_indicator(my_time_series, column='close', output_name='RSI', lower_barrier=35, upper_barrier=65):
    my_time_series = rsi(my_time_series, column, output_name, rsi_lookback=14)
    my_time_series = slope(my_time_series, source='RSI', output_name='slope_rsi', slope_lookback=14)
    my_time_series = slope(my_time_series, source='close', output_name='slope_market', slope_lookback=14)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['slope_rsi'].iloc[i] > 0 and \
           my_time_series['slope_rsi'].iloc[i-1] < 0 and \
           my_time_series['slope_market'].iloc[i] < 0 and \
           my_time_series['slope_market'].iloc[i-1] < 0 and \
           my_time_series['RSI'].iloc[i] < lower_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['slope_rsi'].iloc[i] < 0 and \
             my_time_series['slope_rsi'].iloc[i-1] > 0 and \
             my_time_series['slope_market'].iloc[i] > 0 and \
             my_time_series['slope_market'].iloc[i-1] > 0 and \
             my_time_series['RSI'].iloc[i] > upper_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def green_indicator(my_time_series, column='close', output_name='RSI', rsi_lookback=14, slope_lookback=14,
                     lower_barrier=35, upper_barrier=65):
    my_time_series = rsi(my_time_series, source='close', output_name='RSI', rsi_lookback=rsi_lookback)
    my_time_series = slope(my_time_series, source='RSI', output_name='slope', slope_lookback=slope_lookback)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['slope'].iloc[i] > 0 and \
           my_time_series['slope'].iloc[i-1] < 0 and \
           my_time_series['RSI'].iloc[i] < lower_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['slope'].iloc[i] < 0 and \
             my_time_series['slope'].iloc[i-1] > 0 and \
             my_time_series['RSI'].iloc[i] > upper_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def blue_indicator(my_time_series, column='close', output_name='RSI', rsi_lookback=5, slope_lookback=5,
                    lower_barrier=30, upper_barrier=70, margin=5):
    my_time_series = slope(my_time_series, source='close', output_name='slope', slope_lookback=slope_lookback)
    my_time_series = rsi(my_time_series, source='slope', output_name='RSI_slope', rsi_lookback=rsi_lookback)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['RSI_slope'].iloc[i] > lower_barrier and \
           my_time_series['RSI_slope'].iloc[i] < lower_barrier + margin and \
           my_time_series['RSI_slope'].iloc[i-1] < lower_barrier and \
           my_time_series['low'].iloc[i] < my_time_series['low'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['RSI_slope'].iloc[i] < upper_barrier and \
             my_time_series['RSI_slope'].iloc[i] > upper_barrier - margin and \
             my_time_series['RSI_slope'].iloc[i-1] > upper_barrier and \
             my_time_series['high'].iloc[i] > my_time_series['high'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def indigo_indicator(my_time_series, source='close'):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series[source].iloc[i] > my_time_series[source].iloc[i-1] and \
           my_time_series[source].iloc[i-1] < my_time_series[source].iloc[i-2] and \
           my_time_series[source].iloc[i-2] < my_time_series[source].iloc[i-3] and \
           my_time_series[source].iloc[i-3] < my_time_series[source].iloc[i-5] and \
           my_time_series[source].iloc[i-5] < my_time_series[source].iloc[i-8] and \
           my_time_series[source].iloc[i-8] < my_time_series[source].iloc[i-13] and \
           my_time_series[source].iloc[i-13] < my_time_series[source].iloc[i-21] and \
           my_time_series[source].iloc[i-21] < my_time_series[source].iloc[i-34]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series[source].iloc[i] < my_time_series[source].iloc[i-1] and \
             my_time_series[source].iloc[i-1] > my_time_series[source].iloc[i-2] and \
             my_time_series[source].iloc[i-2] > my_time_series[source].iloc[i-3] and \
             my_time_series[source].iloc[i-3] > my_time_series[source].iloc[i-5] and \
             my_time_series[source].iloc[i-5] > my_time_series[source].iloc[i-8] and \
             my_time_series[source].iloc[i-8] > my_time_series[source].iloc[i-13] and \
             my_time_series[source].iloc[i-13] > my_time_series[source].iloc[i-21] and \
             my_time_series[source].iloc[i-21] > my_time_series[source].iloc[i-34]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def violet_indicator(my_time_series, source='close', ma_source='HMA'):
    my_time_series = hma(my_time_series, ma_lookback=20)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series[source].iloc[i] > my_time_series[ma_source].iloc[i] and \
           my_time_series[source].iloc[i-1] < my_time_series[ma_source].iloc[i-1] and \
           my_time_series[source].iloc[i-2] < my_time_series[ma_source].iloc[i-2] and \
           my_time_series[source].iloc[i-3] < my_time_series[ma_source].iloc[i-3] and \
           my_time_series[source].iloc[i-5] < my_time_series[ma_source].iloc[i-5] and \
           my_time_series[source].iloc[i-8] < my_time_series[ma_source].iloc[i-8] and \
           my_time_series[source].iloc[i-13] < my_time_series[ma_source].iloc[i-13] and \
           my_time_series[source].iloc[i-21] < my_time_series[ma_source].iloc[i-21]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series[source].iloc[i] < my_time_series[ma_source].iloc[i] and \
             my_time_series[source].iloc[i-1] > my_time_series[ma_source].iloc[i-1] and \
             my_time_series[source].iloc[i-2] > my_time_series[ma_source].iloc[i-2] and \
             my_time_series[source].iloc[i-3] > my_time_series[ma_source].iloc[i-3] and \
             my_time_series[source].iloc[i-5] > my_time_series[ma_source].iloc[i-5] and \
             my_time_series[source].iloc[i-8] > my_time_series[ma_source].iloc[i-8] and \
             my_time_series[source].iloc[i-13] > my_time_series[ma_source].iloc[i-13] and \
             my_time_series[source].iloc[i-21] > my_time_series[ma_source].iloc[i-21]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

def kama(my_time_series, source='close', ma_lookback=50, fastest=2, slowest=30):
    def kaufman_er(my_time_series, source='close', er_lookback=20):
        change = abs(my_time_series[source].diff(er_lookback))
        volatility = my_time_series[source].diff().abs().rolling(window=er_lookback).sum()
        er = change / volatility
        return er
    # calculate the ER using the previous function
    er = kaufman_er(my_time_series, source='close', er_lookback=20)
    # calculate the SC
    fastest_sc = 2 / (fastest + 1)
    slowest_sc = 2 / (slowest + 1)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    # initialize KAMA series with NaN
    my_time_series['KAMA'] = pd.Series(np.nan, index=my_time_series.index)
    # start KAMA calculation from the first available data point
    my_time_series['KAMA'].iloc[ma_lookback] = my_time_series[source].iloc[ma_lookback]
    # calculate KAMA iteratively
    for i in range(ma_lookback + 1, len(my_time_series[source])):
        my_time_series['KAMA'].iloc[i] = my_time_series['KAMA'].iloc[i-1] + \
                                         sc.iloc[i] * (my_time_series[source].iloc[i] - \
                                         my_time_series['KAMA'].iloc[i-1])
    return my_time_series.dropna()

def alma(my_time_series, source='close', ma_lookback=50, offset=0.85, sigma=3):
    # initialize the weights array
    m = offset * (ma_lookback - 1)
    s = ma_lookback / sigma
    weights = np.exp(-((np.arange(ma_lookback) - m) ** 2) / (2 * s ** 2))
    weights /= np.sum(weights)
    # calculate ALMA
    my_time_series['ALMA'] = ''
    my_time_series['ALMA'] = my_time_series[source].rolling(ma_lookback).apply(lambda x: np.dot(x, weights), raw=True)
    return my_time_series.dropna()

def lsma(my_time_series, ma_lookback=50):
    my_time_series_copy = np.asarray(my_time_series)
    n = len(my_time_series_copy)
    lsma = np.full(n, np.nan)
    x = np.arange(1, ma_lookback+1)
    X = np.vstack([np.ones(ma_lookback), x]).T
    for t in range(ma_lookback - 1, n):
        y_window = my_time_series_copy[t-ma_lookback + 1 : t+1, 3]
        beta, *_ = np.linalg.lstsq(X, y_window, rcond=None)
        lsma_value = beta[0] + beta[1] * ma_lookback
        lsma[t] = lsma_value
    lsma = pd.DataFrame(lsma)
    my_time_series['LSMA'] = lsma.values
    return my_time_series.dropna()

def volume_candlestick_plot(my_time_series, window=250):
    sample = my_time_series.iloc[-window:, ] 
    fig, ax = plt.subplots(figsize = (10, 5))    
    for i in sample.index:  
        plt.vlines(x = i, ymin = sample.at[i, 'low'], ymax = sample.at[i, 'high'], color = 'black', linewidth = 1)  
        if sample.at[i, 'close'] > sample.at[i, 'open'] and sample.at[i, 'normalized'] >= 0.75: 
            plt.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'green', linewidth = 7) 
        elif sample.at[i, 'close'] > sample.at[i, 'open'] and sample.at[i, 'normalized'] < 0.75 and sample.at[i, 'normalized'] >= 0.50:  
            plt.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'green', linewidth = 5) 
        elif sample.at[i, 'close'] > sample.at[i, 'open'] and sample.at[i, 'normalized'] > 0.25 and sample.at[i, 'normalized'] < 0.50:  
            plt.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'green', linewidth = 3) 
        elif sample.at[i, 'close'] > sample.at[i, 'open'] and sample.at[i, 'normalized'] <= 0.25:  
            plt.vlines(x = i, ymin = sample.at[i, 'open'], ymax = sample.at[i, 'close'], color = 'green', linewidth = 2) 
        elif sample.at[i, 'close'] < sample.at[i, 'open'] and sample.at[i, 'normalized'] >= 0.75: 
            plt.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'red', linewidth = 7) 
        elif sample.at[i, 'close'] < sample.at[i, 'open'] and sample.at[i, 'normalized'] < 0.75 and sample.at[i, 'normalized'] >= 0.50:  
            plt.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'red', linewidth = 5) 
        elif sample.at[i, 'close'] < sample.at[i, 'open'] and sample.at[i, 'normalized'] > 0.25 and sample.at[i, 'normalized'] < 0.50:  
            plt.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'red', linewidth = 3) 
        elif sample.at[i, 'close'] < sample.at[i, 'open'] and sample.at[i, 'normalized'] <= 0.25:  
            plt.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'red', linewidth = 2)                
        elif sample.at[i, 'close'] == sample.at[i, 'open']:
            plt.vlines(x = i, ymin = sample.at[i, 'close'], ymax = sample.at[i, 'open'], color = 'black', linewidth = 1)  
    plt.grid()
    plt.show()
    plt.tight_layout()
    
def ha_plot(my_time_series, window):
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ] 
    # create a double plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # plot heikin-ashi
    for i in sample.index:  
        ax1.vlines(x=i, ymin=sample.at[i, 'HA_low'], ymax=sample.at[i, 'HA_high'], color='black', linewidth=1)  
        if sample.at[i, 'HA_close'] > sample.at[i, 'HA_open']: 
            ax1.vlines(x=i, ymin=sample.at[i, 'HA_open'], ymax=sample.at[i, 'HA_close'], color='green', linewidth=3)  
        if sample.at[i, 'HA_close'] < sample.at[i, 'HA_open']:
            ax1.vlines(x=i, ymin=sample.at[i, 'HA_close'], ymax=sample.at[i, 'HA_open'], color='red', linewidth=3)   
        if sample.at[i, 'HA_close'] == sample.at[i, 'HA_open']:
            ax1.vlines(x=i, ymin=sample.at[i, 'HA_close'], ymax=sample.at[i, 'HA_open'], color='black', linewidth=5)  
    ax1.grid()
    ax1.set_title('Heikin-Ashi Candlestick Chart')
    # plot regular candlesticks
    for i in sample.index:  
        ax2.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
        if sample.at[i, 'close'] > sample.at[i, 'open']: 
            ax2.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
        if sample.at[i, 'close'] < sample.at[i, 'open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
        if sample.at[i, 'close'] == sample.at[i, 'open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=5)  
    ax2.grid()
    ax2.set_title('Regular Candlestick Chart')
    plt.tight_layout()
    
def k_plot(my_time_series, window):
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ] 
    # create a double plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # plot k's candlesticks
    for i in sample.index:  
        ax1.vlines(x=i, ymin=sample.at[i, 'k_low'], ymax=sample.at[i, 'k_high'], color='black', linewidth=1)  
        if sample.at[i, 'k_close'] > sample.at[i, 'k_open']: 
            ax1.vlines(x=i, ymin=sample.at[i, 'k_open'], ymax=sample.at[i, 'k_close'], color='green', linewidth=3)  
        if sample.at[i, 'k_close'] < sample.at[i, 'k_open']:
            ax1.vlines(x=i, ymin=sample.at[i, 'k_close'], ymax=sample.at[i, 'k_open'], color='red', linewidth=3)   
        if sample.at[i, 'k_close'] == sample.at[i, 'k_open']:
            ax1.vlines(x=i, ymin=sample.at[i, 'k_close'], ymax=sample.at[i, 'k_open'], color='black', linewidth=5)  
    ax1.grid()
    ax1.set_title('K`s Candlestick Chart')
    # plot regular candlesticks
    for i in sample.index:  
        ax2.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
        if sample.at[i, 'close'] > sample.at[i, 'open']: 
            ax2.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
        if sample.at[i, 'close'] < sample.at[i, 'open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
        if sample.at[i, 'close'] == sample.at[i, 'open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=5)  
    ax2.grid()
    ax2.set_title('Regular Candlestick Chart')
    plt.tight_layout()
    
def detect_divergences(my_time_series, lower_barrier=30, upper_barrier=70, width=20):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(len(my_time_series)): 
        try:
            if my_time_series['RSI'].iloc[i] < lower_barrier and my_time_series['RSI'].iloc[i-1] > lower_barrier:
                for a in range(i+1, i+width):
                    if my_time_series['RSI'].iloc[a] > lower_barrier:
                        for r in range(a+1, a+width):
                           if my_time_series['RSI'].iloc[r] < lower_barrier and \
                              my_time_series['RSI'].iloc[r] >= my_time_series['RSI'].iloc[i:r].min() and \
                              my_time_series['close'].iloc[r] <= my_time_series['close'].iloc[i:r].min():
                                for s in range(r+1, r+width): 
                                    if my_time_series['RSI'].iloc[s] > lower_barrier:
                                        my_time_series['bullish_signal'].loc[s+1] = 1
                                        break
                                    else:
                                        continue
                           else:
                                continue
                        else:
                            continue
                    else:
                        continue             
        except IndexError:
              pass
    for i in range(len(my_time_series)): 
        try:
            if my_time_series['RSI'].iloc[i] > upper_barrier and my_time_series['RSI'].iloc[i-1] < upper_barrier:
                for a in range(i+1, i+width):
                    if my_time_series['RSI'].iloc[a] < upper_barrier:
                        for r in range(a+1, a+width):
                           if my_time_series['RSI'].iloc[r] > upper_barrier and \
                              my_time_series['RSI'].iloc[r] <= my_time_series['RSI'].iloc[i:r].max() and \
                              my_time_series['close'].iloc[r] >= my_time_series['close'].iloc[i:r].max():
                                for s in range(r+1, r+width): 
                                    if my_time_series['RSI'].iloc[s] < upper_barrier:
                                        my_time_series['bearish_signal'].iloc[s+1] = 1
                                        break
                                    else:
                                        continue
                           else:
                                continue
                        else:
                            continue
                    else:
                        continue             
        except IndexError:
              pass
    return my_time_series

def detect_double_top_bottom(my_time_series, swing_lookback=20, tolerance=0.05):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    swings = []
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
        elif not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
    for i in range(len(swings) - 2):
        idx1, type1, val1 = swings[i]
        idx2, type2, val2 = swings[i+1]
        idx3, type3, val3 = swings[i+2]
        if type1 == 'low' and type2 == 'high' and type3 == 'low':
            if val3 >= val1 and val3 <= val1 * (1 + tolerance):
                neckline = val2
                for j in range(idx3 + 1, len(my_time_series)):
                    if not pd.isna(my_time_series.loc[j, 'swing_low']):
                        break
                    elif my_time_series.loc[j, 'close'] > neckline:
                        my_time_series.loc[j+1, 'bullish_signal'] = 1
                        break
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings) - 2):
        idx1, type1, val1 = swings[i]
        idx2, type2, val2 = swings[i+1]
        idx3, type3, val3 = swings[i+2]
        if type1 == 'high' and type2 == 'low' and type3 == 'high':
            if val3 <= val1 and val3 >= val1 * (1 - tolerance):
                neckline = val2  # the swing low between the tops
                for j in range(idx3 + 1, len(my_time_series)):
                    if not pd.isna(my_time_series.loc[j, 'swing_high']):
                        break
                    elif my_time_series.loc[j, 'close'] < neckline:
                        my_time_series.loc[j+1, 'bearish_signal'] = 1
                        break
    return my_time_series

def detect_head_and_shoulders(my_time_series, swing_lookback=20, tolerance=0.05):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    swings = []
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings)-4):
        idx1, type1, val1 = swings[i] 
        idx2, type2, val2 = swings[i+1]
        idx3, type3, val3 = swings[i+2]
        idx4, type4, val4 = swings[i+3]
        idx5, type5, val5 = swings[i+4]
        if (type1 == 'low' and type2 == 'high' and
            type3 == 'low' and type4 == 'high' and
            type5 == 'low'):
            if val3 < val1 and val3 < val5:
                shoulders_close = abs(val1 - val5) <= val1 * tolerance
                peaks_close = abs(val2 - val4) <= val2 * tolerance
                if shoulders_close and peaks_close:
                    neckline = (val2 + val4) / 2
                    for j in range(idx5 + 1, len(my_time_series)):
                        if not pd.isna(my_time_series.loc[j, 'swing_low']):
                            break
                        elif my_time_series.loc[j, 'close'] > neckline:
                            my_time_series.loc[j+1, 'bullish_signal'] = 1
                            break
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings)-4):
        idx1, type1, val1 = swings[i]
        idx2, type2, val2 = swings[i+1]
        idx3, type3, val3 = swings[i+2]
        idx4, type4, val4 = swings[i+3]
        idx5, type5, val5 = swings[i+4]
        if (type1 == 'high' and type2 == 'low' and
            type3 == 'high' and type4 == 'low' and
            type5 == 'high'):
            if val3 > val1 and val3 > val5:
                shoulders_close = abs(val1 - val5) <= val1 * tolerance
                troughs_close = abs(val2 - val4) <= val2 * tolerance
                if shoulders_close and troughs_close:
                    neckline = (val2 + val4) / 2
                    for j in range(idx5 + 1, len(my_time_series)):
                        if not pd.isna(my_time_series.loc[j, 'swing_high']):
                            break
                        elif my_time_series.loc[j, 'close'] < neckline:
                            my_time_series.loc[j+1, 'bearish_signal'] = 1
                            break
    return my_time_series

def detect_gartley_pattern(my_time_series, swing_lookback=20, fib_tolerance=3):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    swings = []
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings) - 4):
        p0, p1, p2, p3, p4 = swings[i:i+5]
        idx_x, type_x, px = p0
        idx_a, type_a, pa = p1
        idx_b, type_b, pb = p2
        idx_c, type_c, pc = p3
        idx_d, type_d, pd_ = p4
        if [type_x, type_a, type_b, type_c, type_d] == ['low', 'high', 'low', 'high', 'low']:
            xa = pa - px
            ab = pb - pa
            bc = pc - pb
            cd = pd_ - pc
            ad = pd_ - px
            if np.isclose(abs(ab) / abs(xa), 0.618, atol=fib_tolerance) and \
               0.382 <= abs(bc) / abs(ab) <= 0.886 and \
               1.27 <= abs(cd) / abs(bc) <= 1.618 and \
               np.isclose(abs(ad) / abs(xa), 0.786, atol=fib_tolerance) and \
               pd_ >= px:
                my_time_series.loc[idx_d+1, 'bullish_signal'] = 1
        elif [type_x, type_a, type_b, type_c, type_d] == ['high', 'low', 'high', 'low', 'high']:
            xa = px - pa
            ab = pa - pb
            bc = pb - pc
            cd = pc - pd_
            ad = px - pd_
            if np.isclose(abs(ab) / abs(xa), 0.618, atol=fib_tolerance) and \
               0.382 <= abs(bc) / abs(ab) <= 0.886 and \
               1.27 <= abs(cd) / abs(bc) <= 1.618 and \
               np.isclose(abs(ad) / abs(xa), 0.786, atol=fib_tolerance) and \
               pd_ <= px:
                my_time_series.loc[idx_d+1, 'bearish_signal'] = 1
    return my_time_series

def detect_crab_pattern(my_time_series, swing_lookback=20, fib_tolerance=3):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    swings = []
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings)-4):
        p0, p1, p2, p3, p4 = swings[i:i+5]
        idx_x, type_x, px = p0
        idx_a, type_a, pa = p1
        idx_b, type_b, pb = p2
        idx_c, type_c, pc = p3
        idx_d, type_d, pd_ = p4
        if [type_x, type_a, type_b, type_c, type_d] == ['low', 'high', 'low', 'high', 'low']:
            xa = pa - px
            ab = pb - pa
            bc = pc - pb
            cd = pd_ - pc
            ad = pd_ - px
            if np.isclose(abs(ab) / abs(xa), 0.382, atol=fib_tolerance) and \
               0.382 <= abs(bc) / abs(ab) <= 0.886 and \
               1.618 <= abs(cd) / abs(bc) <= 3.618 and \
               np.isclose(abs(ad) / abs(xa), 1.618, atol=fib_tolerance) and \
               pd_ <= px:
                my_time_series.loc[idx_d+1, 'bullish_signal'] = 1
        elif [type_x, type_a, type_b, type_c, type_d] == ['high', 'low', 'high', 'low', 'high']:
            xa = px - pa
            ab = pa - pb
            bc = pb - pc
            cd = pc - pd_
            ad = px - pd_
            if np.isclose(abs(ab) / abs(xa), 0.382, atol=fib_tolerance) and \
               0.382 <= abs(bc) / abs(ab) <= 0.886 and \
               1.618 <= abs(cd) / abs(bc) <= 3.618 and \
               np.isclose(abs(ad) / abs(xa), 1.618, atol=fib_tolerance) and \
               pd_ >= px:
                my_time_series.loc[idx_d+1, 'bearish_signal'] = 1
    return my_time_series

def detect_failed_extreme_xabc(my_time_series, swing_lookback=20, fib_tolerance=0.05):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series = swing_detect(my_time_series, swing_lookback=swing_lookback)
    swings = []
    for idx, row in my_time_series.iterrows():
        if not pd.isna(row['swing_high']):
            swings.append((idx, 'high', row['swing_high']))
        elif not pd.isna(row['swing_low']):
            swings.append((idx, 'low', row['swing_low']))
    for i in range(len(swings)-3):
        p0, p1, p2, p3 = swings[i:i+4]
        idx_x, type_x, px = p0
        idx_a, type_a, pa = p1
        idx_b, type_b, pb = p2
        idx_c, type_c, pc = p3
        if [type_x, type_a, type_b, type_c] == ['high', 'low', 'high', 'low']:
            xa = px - pa
            ab = pb - pa
            bc = pb - pc
            ab_ratio = abs(ab) / abs(xa)
            bc_ratio = abs(bc) / abs(ab)
            if (1.13 - fib_tolerance) <= ab_ratio <= (1.618 + fib_tolerance) and \
               (1.618 - fib_tolerance) <= bc_ratio <= (2.24 + fib_tolerance) and \
               pb > px and pc < pa:
                my_time_series.loc[idx_c+1, 'bullish_signal'] = 1
        if [type_x, type_a, type_b, type_c] == ['low', 'high', 'low', 'high']:
            xa = pa - px
            ab = pb - pa
            bc = pc - pb
            ab_ratio = abs(ab) / abs(xa)
            bc_ratio = abs(bc) / abs(ab)
            if (1.13 - fib_tolerance) <= ab_ratio <= (1.618 + fib_tolerance) and \
               (1.618 - fib_tolerance) <= bc_ratio <= (2.24 + fib_tolerance) and \
               pb < px and pc > pa:
                my_time_series.loc[idx_c+1, 'bearish_signal'] = 1
    return my_time_series

def signal_chart_complex_harmonic_pattern(my_time_series, window=500, source='open'): 
    sample = my_time_series.iloc[-window:, ]
    sample['zigzag'] = sample['swing_high'].combine_first(sample['swing_low'])
    zigzag = sample['zigzag'].dropna()
    ohlc_plot(sample, window, plot_type='candlesticks')   
    plt.plot(zigzag.index, zigzag, color='orange', linewidth=2, label='ZigZag', linestyle='dashed')     
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='.', s=50, zorder=3)
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='.', s=50, zorder=3)
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))   
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    swing_point = Line2D([0], [0], marker='.', color='w', label='Swing Points', markerfacecolor='blue', markersize=10) 
    zig_zag = Line2D([0], [0], marker='_', color='orange', label='Zig Zag', markerfacecolor='orange', markersize=10)     
    plt.legend(handles=[bullish_signal, bearish_signal, swing_point, zig_zag])
    plt.tight_layout()
    
def signal_chart_price_pattern(my_time_series, window=500, source='open'): 
    sample = my_time_series.iloc[-window:, ]
    sample['zigzag'] = sample['swing_high'].combine_first(sample['swing_low'])
    ohlc_plot(sample, window, plot_type='bars')   
    plt.scatter(sample.index, sample['swing_high'], color='blue', marker='.', s=50, zorder=3)
    plt.scatter(sample.index, sample['swing_low'], color='blue', marker='.', s=50, zorder=3)
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source]+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))   
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    swing_point = Line2D([0], [0], marker='.', color='w', label='Swing Points', markerfacecolor='blue', markersize=10) 
    plt.legend(handles=[bullish_signal, bearish_signal, swing_point])
    plt.tight_layout()
    
def detect_divergences_rsi_square(my_time_series, lower_barrier=40, upper_barrier=60, width=20):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(len(my_time_series)): 
        try:
            if my_time_series['RSI²'].iloc[i] < lower_barrier and my_time_series['RSI²'].iloc[i-1] > lower_barrier:
                for a in range(i+1, i+width):
                    if my_time_series['RSI²'].iloc[a] > lower_barrier:
                        for r in range(a+1, a+width):
                           if my_time_series['RSI²'].iloc[r] < lower_barrier and \
                              my_time_series['RSI²'].iloc[r] >= my_time_series['RSI²'].iloc[i:r].min() and \
                              my_time_series['RSI'].iloc[r] <= my_time_series['RSI'].iloc[i:r].min():
                                for s in range(r+1, r+width): 
                                    if my_time_series['RSI²'].iloc[s] > lower_barrier:
                                        my_time_series['bullish_signal'].iloc[s+1] = 1
                                        break
                                    elif my_time_series['RSI²'].iloc[s] < my_time_series['RSI²'].iloc[i:r].min():
                                        break
                                    else:
                                        continue
                           else:
                                continue
                        else:
                            continue
                    else:
                        continue             
        except IndexError:
              pass
    for i in range(len(my_time_series)): 
        try:
            if my_time_series['RSI²'].iloc[i] > upper_barrier and my_time_series['RSI²'].iloc[i-1] < upper_barrier:
                for a in range(i+1, i+width):
                    if my_time_series['RSI²'].iloc[a] < upper_barrier:
                        for r in range(a+1, a+width):
                           if my_time_series['RSI²'].iloc[r] > upper_barrier and \
                              my_time_series['RSI²'].iloc[r] <= my_time_series['RSI²'].iloc[i:r].max() and \
                              my_time_series['RSI'].iloc[r] >= my_time_series['RSI'].iloc[i:r].max():
                                for s in range(r+1, r+width): 
                                    if my_time_series['RSI²'].iloc[s] < upper_barrier:
                                        my_time_series['bearish_signal'].iloc[s+1] = 1
                                        break
                                    elif my_time_series['RSI²'].iloc[s] > my_time_series['RSI²'].iloc[i:r].max():
                                        break
                                    else:
                                        continue
                           else:
                                continue
                        else:
                            continue
                    else:
                        continue             
        except IndexError:
              pass
    return my_time_series

def signal_chart_indicator_rsi_square(my_time_series, window=500, lower_barrier=40, upper_barrier=60, plot_type='bars',
                                      barriers=True): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]   
    if plot_type == 'bars':
        for i in sample.index:  
            ax1.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin = sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)   
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            ax1.vlines(x=i, ymin =sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=5)  
    elif plot_type == 'line':
        ax1.plot(sample['open'], color='black') 
    else:
        print('Choose between bars or candlesticks')           
    for i in my_time_series.index:
        if my_time_series.loc[i, 'bullish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']-1),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        elif my_time_series.loc[i, 'bearish_signal'] == 1:
            ax1.annotate('', xy=(i, my_time_series.loc[i, 'open']), xytext=(i, my_time_series.loc[i, 'open']+1),
                        arrowprops=dict(facecolor='red', shrink=0.05))    
    ax2.plot(sample.index, sample['RSI'], label='RSI', color='blue')
    ax2.plot(sample.index, sample['RSI²'], label='RSI²', color='red', linestyle='dashed')
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='green', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='red', markersize=10)
    ax1.legend(handles=[bullish_signal, bearish_signal])
    if barriers == True:
        ax2.axhline(y=lower_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=upper_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=50, color='black', linestyle='dashed')        
    plt.tight_layout()
    ax1.grid()
    ax2.legend()
    ax2.grid()
    
def gap(my_time_series, atr_column, min_size):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series = atr(my_time_series)
    for i in range(0, len(my_time_series)):
        # bullish gap
        if my_time_series['open'].iloc[i] < my_time_series['close'].iloc[i - 1] and \
           (my_time_series['close'].iloc[i - 1] - my_time_series['open'].iloc[i]) > \
           (my_time_series['volatility'].iloc[i - 1] * min_size):
            my_time_series.at[my_time_series.index[i], 'bullish_signal'] = 1
        # bearish gap
        elif my_time_series['open'].iloc[i] > my_time_series['close'].iloc[i - 1] and \
             (my_time_series['open'].iloc[i] - my_time_series['close'].iloc[i - 1]) > \
             (my_time_series['volatility'].iloc[i - 1] * min_size):
            my_time_series.at[my_time_series.index[i], 'bearish_signal'] = 1
    return my_time_series

def marsi(my_time_series, ma_lookback=200, rsi_lookback=20):
    my_time_series = moving_average(my_time_series, 'close', ma_lookback=ma_lookback)    
    my_time_series = rsi(my_time_series, 'moving_average', 'MARSI', rsi_lookback=rsi_lookback)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):   
        try:
                # bullish signal
                if my_time_series['MARSI'].iloc[i] > 2 and \
                   my_time_series['MARSI'].iloc[i-1] < 2 and \
                   my_time_series['MARSI'].iloc[i-2] < 2 and \
                   my_time_series['MARSI'].iloc[i-3] < 2:
                       my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
                # bearish signal
                elif my_time_series['MARSI'].iloc[i] < 98 and \
                     my_time_series['MARSI'].iloc[i-1] > 98 and \
                     my_time_series['MARSI'].iloc[i-2] > 98 and \
                     my_time_series['MARSI'].iloc[i-3] > 98:
                       my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1 
        except KeyError:
            pass
    return my_time_series 

def fibonacci_moving_average(my_time_series):
    my_time_series['fma_high'] = (my_time_series['high'].ewm(span=2, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=3, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=5, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=8, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=13, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=21, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=34, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=55, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=89, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=144, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=233, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=377, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=610, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=987, adjust=False).mean() + \
                                 my_time_series['high'].ewm(span=1597, adjust=False).mean()) / 15
    my_time_series['fma_low'] = (my_time_series['low'].ewm(span=2, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=3, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=5, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=8, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=13, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=21, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=34, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=55, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=89, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=144, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=233, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=377, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=610, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=987, adjust=False).mean() + \
                                 my_time_series['low'].ewm(span=1597, adjust=False).mean()) / 15
    return my_time_series.dropna()

def rob_booker_reversal(my_time_series):
    my_time_series = stochastic_oscillator(my_time_series, k_lookback=70, k_smoothing_lookback=10, d_lookback=10)
    my_time_series = macd(my_time_series, source='close')   
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['MACD_line'].iloc[i] > 0 and my_time_series['MACD_line'].iloc[i-1] < 0 and \
           my_time_series['%K_smoothing'].iloc[i] < 30:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['MACD_line'].iloc[i] < 0 and my_time_series['MACD_line'].iloc[i-1] > 0 and \
             my_time_series['%K_smoothing'].iloc[i] > 70:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
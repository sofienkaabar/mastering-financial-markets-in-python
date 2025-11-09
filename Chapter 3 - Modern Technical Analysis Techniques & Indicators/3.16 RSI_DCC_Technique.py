from master_library import import_data, signal_chart, rsi

def rsi_dcc_technique(my_time_series):
    my_time_series = rsi(my_time_series, 'close', 'first_RSI', rsi_lookback=13)
    my_time_series = rsi(my_time_series, 'close', 'second_RSI', rsi_lookback=34)    
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['first_RSI'].iloc[i] > 30 and \
           my_time_series['first_RSI'].iloc[i-1] < 30 and \
           my_time_series['second_RSI'].iloc[i] > 30 and \
           my_time_series['second_RSI'].iloc[i-1] < 30:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['first_RSI'].iloc[i] < 70 and \
             my_time_series['first_RSI'].iloc[i-1] > 70 and \
             my_time_series['second_RSI'].iloc[i] < 70 and \
             my_time_series['second_RSI'].iloc[i-1] > 70:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
    
# download the historical values
my_time_series = import_data('DIS')
# call the function(s)
my_time_series = rsi_dcc_technique(my_time_series)
# plot
signal_chart(my_time_series, window=1000, choice='bars', source='open', chart_type='ohlc')
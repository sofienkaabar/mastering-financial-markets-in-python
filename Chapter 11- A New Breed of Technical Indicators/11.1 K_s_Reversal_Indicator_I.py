from master_library import import_data, signal_chart, bollinger_bands, macd

def k_reversal_indicator_I(my_time_series):
    my_time_series = macd(my_time_series, source='close', short_window=12, long_window=26, signal_window=9)
    my_time_series = bollinger_bands(my_time_series, source='close', bb_lookback=100, num_std_dev=2)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['low'].iloc[i] < my_time_series['lower_band'].iloc[i] and \
           my_time_series['high'].iloc[i] < my_time_series['middle_band'].iloc[i] and \
           my_time_series['MACD_line'].iloc[i] > my_time_series['MACD_signal'].iloc[i] and \
           my_time_series['MACD_line'].iloc[i-1] < my_time_series['MACD_signal'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['high'].iloc[i] > my_time_series['upper_band'].iloc[i] and \
           my_time_series['low'].iloc[i] > my_time_series['middle_band'].iloc[i] and \
           my_time_series['MACD_line'].iloc[i] < my_time_series['MACD_signal'].iloc[i] and \
           my_time_series['MACD_line'].iloc[i-1] > my_time_series['MACD_signal'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
    
# download the historical values
my_time_series = import_data(name='PEP')
# call the function(s)
my_time_series = k_reversal_indicator_I(my_time_series)
# plot
signal_chart(my_time_series, 1000, 'bars')

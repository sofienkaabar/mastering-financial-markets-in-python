from master_library import import_data, signal_chart

def extreme_euphoria(my_time_series):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series['absolute_range'] = abs(my_time_series['close'] - my_time_series['open'])
    for i in range(0, len(my_time_series)):  
        try:
            # bullish signal
            if my_time_series['close'].iloc[i] < my_time_series['open'].iloc[i] and \
               my_time_series['close'].iloc[i-1] < my_time_series['open'].iloc[i-1] and \
               my_time_series['close'].iloc[i-2] < my_time_series['open'].iloc[i-2] and \
               my_time_series['close'].iloc[i-3] < my_time_series['open'].iloc[i-3] and \
               my_time_series['close'].iloc[i-4] < my_time_series['open'].iloc[i-4] and \
               my_time_series['absolute_range'].iloc[i] > my_time_series['absolute_range'].iloc[i-1] and \
               my_time_series['absolute_range'].iloc[i-1] > my_time_series['absolute_range'].iloc[i-2]:
                   my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
            # bearish signal
            elif my_time_series['close'].iloc[i] > my_time_series['open'].iloc[i] and \
                 my_time_series['close'].iloc[i-1] > my_time_series['open'].iloc[i-1] and \
                 my_time_series['close'].iloc[i-2] > my_time_series['open'].iloc[i-2] and \
                 my_time_series['close'].iloc[i-3] > my_time_series['open'].iloc[i-3] and \
                 my_time_series['close'].iloc[i-4] > my_time_series['open'].iloc[i-4] and \
                 my_time_series['absolute_range'].iloc[i] > my_time_series['absolute_range'].iloc[i-1] and \
                 my_time_series['absolute_range'].iloc[i-1] > my_time_series['absolute_range'].iloc[i-2]:
                   my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1 
        except KeyError: 
            pass                  
    return my_time_series  

# download the historical values
my_time_series = import_data(name='PYPL')
# call the function(s)
my_time_series = extreme_euphoria(my_time_series)
# plot
signal_chart(my_time_series, 400, 'candlesticks')
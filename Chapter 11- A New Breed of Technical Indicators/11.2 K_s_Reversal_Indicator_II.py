from master_library import import_data, signal_chart, moving_average

def k_reversal_indicator_II(my_time_series):
    my_time_series = moving_average(my_time_series, 
                                    source='close', 
                                    ma_lookback=13, 
                                    output_name='moving_average', 
                                    ma_type='SMA')
    my_time_series['above_SMA'] = (my_time_series['close'] > my_time_series['moving_average']).astype(int)
    my_time_series['rolling_sum'] = my_time_series['above_SMA'].rolling(window=21).sum()
    my_time_series['percentage'] = my_time_series['rolling_sum'] / 21 * 100
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['percentage'].iloc[i] == 0 and my_time_series['percentage'].iloc[i-1] > 0:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['percentage'].iloc[i] == 100 and my_time_series['percentage'].iloc[i-1] < 100:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    my_time_series = my_time_series.drop(columns=['moving_average', 
                                                  'above_SMA', 
                                                  'rolling_sum', 
                                                  'percentage'])
    return my_time_series
    
# download the historical values
my_time_series = import_data(name='PEP')
# call the function(s)
my_time_series = k_reversal_indicator_II(my_time_series)
# plot
signal_chart(my_time_series, 1000, 'bars')

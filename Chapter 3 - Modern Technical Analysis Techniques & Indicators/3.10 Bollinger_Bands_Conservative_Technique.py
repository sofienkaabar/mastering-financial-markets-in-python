import matplotlib.pyplot as plt
from master_library import import_data, bollinger_bands, signal_chart

def bb_conservative_technique(my_time_series, bb_lookback=20, num_std_dev=2):
    my_time_series = bollinger_bands(my_time_series, source='close', bb_lookback=bb_lookback, num_std_dev=num_std_dev)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['close'].iloc[i] > my_time_series['lower_band'].iloc[i] and \
           my_time_series['close'].iloc[i] < my_time_series['middle_band'].iloc[i] and \
           my_time_series['close'].iloc[i-1] < my_time_series['lower_band'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['close'].iloc[i] < my_time_series['upper_band'].iloc[i] and \
             my_time_series['close'].iloc[i] > my_time_series['middle_band'].iloc[i] and \
             my_time_series['close'].iloc[i-1] > my_time_series['upper_band'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
    
# download the historical values
my_time_series = import_data('JPM')
# call the function(s)
my_time_series = bb_conservative_technique(my_time_series, bb_lookback=20, num_std_dev=2)
# plot
signal_chart(my_time_series, 250)
plt.plot(my_time_series['lower_band'].iloc[-250:], label='Lower Band', color='darkgrey')
plt.plot(my_time_series['upper_band'].iloc[-250:], label='Upper Band', color='black')
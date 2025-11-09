import matplotlib.pyplot as plt
from master_library import import_data, moving_average, signal_chart, performance_evaluation

def moving_average_cross_strategy(my_time_series, short_moving_average=30, long_moving_average=50):
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    my_time_series = moving_average(my_time_series, source='close', ma_lookback=short_moving_average, output_name='short_moving_average', ma_type='SMA')
    my_time_series = moving_average(my_time_series, source='close', ma_lookback=long_moving_average, output_name='long_moving_average', ma_type='SMA')
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['short_moving_average'].iloc[i] > my_time_series['long_moving_average'].iloc[i] and \
           my_time_series['short_moving_average'].iloc[i-1] < my_time_series['long_moving_average'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['short_moving_average'].iloc[i] < my_time_series['long_moving_average'].iloc[i] and \
             my_time_series['short_moving_average'].iloc[i-1] > my_time_series['long_moving_average'].iloc[i-1]:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series

# download the historical values
my_time_series = import_data(name='AAPL')
# call the function(s)
my_time_series = moving_average_cross_strategy(my_time_series)
# plot
signal_chart(my_time_series, window=500, choice='bars', source='open', chart_type='ohlc')
plt.plot(my_time_series['long_moving_average'][-500:])
plt.plot(my_time_series['short_moving_average'][-500:])
# performance evaluation
my_time_series = performance_evaluation(my_time_series, strategy='variable_holding_period', holding_period=10)

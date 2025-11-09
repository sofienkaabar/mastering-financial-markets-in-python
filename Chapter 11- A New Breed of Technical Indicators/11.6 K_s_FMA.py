import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, fibonacci_moving_average

# download the historical values
my_time_series = import_data('BTC-USD')
# call the function(s)
my_time_series = fibonacci_moving_average(my_time_series)
# plot
ohlc_plot(my_time_series, window=500, plot_type='bars', chart_type='ohlc')
plt.plot(my_time_series.iloc[-500:]['fma_high'], color = 'lightblue', label='Fibonacci Moving Average')
plt.plot(my_time_series.iloc[-500:]['fma_low'], color = 'lightblue')
plt.fill_between(my_time_series[-500:].index, 
                 my_time_series.iloc[-500:]['fma_high'], 
                 my_time_series.iloc[-500:]['fma_low'], color='lightblue')
plt.legend()
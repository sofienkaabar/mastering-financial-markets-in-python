import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, moving_average

# download the historical values
my_time_series = import_data('BTC-USD')
# call the function(s)
my_time_series = moving_average(my_time_series, ma_lookback=100, output_name='SMMA', ma_type='SMMA')
# plot
ohlc_plot(my_time_series, 250, 'bars')
plt.plot(my_time_series['SMMA'][-250:, ], label='SMMA')
plt.legend()
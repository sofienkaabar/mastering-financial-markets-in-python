import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, kama

# download the historical values
my_time_series = import_data('BTC-USD')
# call the function(s)
my_time_series = kama(my_time_series, source='close', ma_lookback=50, fastest=2, slowest=30)
# plot
ohlc_plot(my_time_series, 250, 'bars')
plt.plot(my_time_series['KAMA'][-250:, ], label='KAMA')
plt.legend()
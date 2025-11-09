import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, iwma

# download the historical values
my_time_series = import_data('BTC-USD')
# call the function(s)
my_time_series['IWMA'] = iwma(my_time_series)
# plot
ohlc_plot(my_time_series, 250, 'bars')
plt.plot(my_time_series['IWMA'][-250:, ], label='IWMA')
plt.legend()

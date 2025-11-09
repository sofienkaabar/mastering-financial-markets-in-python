import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, alma

# download the historical values
my_time_series = import_data('BTC-USD')
# call the function(s)
my_time_series = alma(my_time_series, source='close')
# plot
ohlc_plot(my_time_series, 250, 'bars')
plt.plot(my_time_series['ALMA'][-250:, ], label='ALMA')
plt.legend()
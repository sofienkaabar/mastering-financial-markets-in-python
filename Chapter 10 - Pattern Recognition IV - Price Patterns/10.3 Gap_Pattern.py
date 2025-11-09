from master_library import import_data, signal_chart, gap

# download the historical values
my_time_series = import_data('AMZN')
# call the function(s)
my_time_series = gap(my_time_series, 'volatility', 1)
# plot
signal_chart(my_time_series, 180, 'candlesticks')
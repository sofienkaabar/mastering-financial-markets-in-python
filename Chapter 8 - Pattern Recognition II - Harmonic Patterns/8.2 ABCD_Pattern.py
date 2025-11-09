from master_library import import_data, abcd_pattern, signal_chart_abcd

# download the historical values
my_time_series = import_data(name='XOM')
# call the function(s)
my_time_series = abcd_pattern(my_time_series, swing_lookback=20)
# plot
signal_chart_abcd(my_time_series, window=1000)

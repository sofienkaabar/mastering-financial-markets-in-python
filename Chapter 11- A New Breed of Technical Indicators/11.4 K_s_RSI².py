from master_library import import_data, rsi_square, detect_divergences_rsi_square, signal_chart_indicator_rsi_square

# download the historical values
my_time_series = import_data('NKE')
# call the function(s)
my_time_series = rsi_square(my_time_series, source='close', rsi_prime_lookback=14, rsi_square_lookback=5)
my_time_series = detect_divergences_rsi_square(my_time_series, lower_barrier=40, upper_barrier=60, width=30)
# plot
signal_chart_indicator_rsi_square(my_time_series, window=500)

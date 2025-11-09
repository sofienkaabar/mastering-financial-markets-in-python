from master_library import import_data, fibonacci_retracement, display_retracement

# download the historical values        
my_time_series = import_data('AMZN')
# call the function(s)
my_time_series = fibonacci_retracement(my_time_series, swing_lookback=20, fib_level=0.618)
# plot
display_retracement(my_time_series, window=250)
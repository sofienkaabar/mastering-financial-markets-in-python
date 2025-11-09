from master_library import import_data, fibonacci_timing_pattern, signal_chart

# download the historical values
my_time_series = import_data(name='TSLA')
# call the function(s)
my_time_series = fibonacci_timing_pattern(my_time_series)
# plot
signal_chart(my_time_series, 500, 'bars')
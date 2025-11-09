from master_library import abcd_pattern, signal_chart_abcd, generate_synthetic_symmetric_data

# generate the historical values
my_time_series = generate_synthetic_symmetric_data(num_data=200, amplitude=5, base_price=100)
# call the function(s)
my_time_series = abcd_pattern(my_time_series, swing_lookback=20)
# plot
signal_chart_abcd(my_time_series, window=100)
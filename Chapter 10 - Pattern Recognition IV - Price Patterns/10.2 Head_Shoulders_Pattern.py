from master_library import detect_head_and_shoulders, generate_ohlc_data, signal_chart_price_pattern

# generate the historical values
my_time_series = generate_ohlc_data(length_data=2000)
# call the function(s)
my_time_series = detect_head_and_shoulders(my_time_series, swing_lookback=60, tolerance=0.05)
# plot
signal_chart_price_pattern(my_time_series, window=1000, source='open')

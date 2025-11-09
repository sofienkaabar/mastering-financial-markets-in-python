from master_library import detect_gartley_pattern, generate_ohlc_data, signal_chart_complex_harmonic_pattern
   
# generate the historical values
my_time_series = generate_ohlc_data(length_data=2000)
# call the function(s)
my_time_series = detect_gartley_pattern(my_time_series, swing_lookback=5, fib_tolerance=1)
# plot
signal_chart_complex_harmonic_pattern(my_time_series, window=200)
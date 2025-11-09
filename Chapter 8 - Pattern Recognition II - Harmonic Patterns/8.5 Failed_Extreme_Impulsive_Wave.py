from master_library import detect_failed_extreme_xabc, generate_ohlc_data, signal_chart_complex_harmonic_pattern
   
# generate the historical values
my_time_series = generate_ohlc_data(length_data=2000)
# call the function(s)
my_time_series = detect_failed_extreme_xabc(my_time_series, swing_lookback=5, fib_tolerance=0.05)
# plot
signal_chart_complex_harmonic_pattern(my_time_series, window=200)

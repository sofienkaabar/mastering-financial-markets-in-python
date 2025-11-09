from master_library import import_data_with_volume, normalize, volume_candlestick_plot
    
# download the historical values
my_time_series = import_data_with_volume('PLTR')
# call the function(s)
my_time_series = normalize(my_time_series, source='volume', output_column='normalized', normalized_lookback=20)
# plot
volume_candlestick_plot(my_time_series, 100)

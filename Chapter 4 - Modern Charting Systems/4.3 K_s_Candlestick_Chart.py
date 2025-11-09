from master_library import import_data, k_candlesticks, k_plot
    
# download the historical values
my_time_series = import_data('META')
# call the function(s)
my_time_series = k_candlesticks(my_time_series, k_lookback=5)
# plot
k_plot(my_time_series, 50)
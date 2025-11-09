from master_library import import_data, swing_detect, swing_points_chart

# download the historical values
my_time_series = import_data('DIS')
# call the function(s)
my_time_series = swing_detect(my_time_series, swing_lookback=20)
# plot
swing_points_chart(my_time_series)
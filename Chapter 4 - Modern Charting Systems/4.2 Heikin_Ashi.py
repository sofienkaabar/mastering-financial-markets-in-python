from master_library import import_data, heikin_ashi, ha_plot

# download the historical values
my_time_series = import_data('DIS')
# call the function(s)
my_time_series = heikin_ashi(my_time_series)
# plot
ha_plot(my_time_series, 50)
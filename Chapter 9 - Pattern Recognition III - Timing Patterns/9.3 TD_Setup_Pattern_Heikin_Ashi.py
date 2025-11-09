from master_library import import_data, signal_chart, heikin_ashi, td_setup

# download the historical values
my_time_series = import_data(name='META')
# call the function(s)
my_time_series = heikin_ashi(my_time_series)
my_time_series = td_setup(my_time_series, 
                 source='HA_close', 
                 perfected_source_low='HA_low', 
                 perfected_source_high='HA_high', 
                 final_step=9, 
                 difference=4, 
                 perfected=True)
# plot
signal_chart(my_time_series, 200, 'candlesticks')
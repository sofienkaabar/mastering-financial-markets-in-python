from master_library import import_data, signal_chart, td_setup, k_candlesticks

# download the historical values
my_time_series = import_data(name='UBER')
# call the function(s)
my_time_series = k_candlesticks(my_time_series, k_lookback=5)
my_time_series = td_setup(my_time_series, 
                 source='k_close', 
                 perfected_source_low='k_low', 
                 perfected_source_high='k_high', 
                 final_step=9, 
                 difference=4, 
                 perfected=True)
# plot
signal_chart(my_time_series, 200, 'candlesticks')

from master_library import import_data, signal_chart, td_setup

# download the historical values
my_time_series = import_data(name='AAPL')
# call the function(s)
my_time_series = td_setup(my_time_series, final_step=9, difference=4, perfected=True)
# plot
signal_chart(my_time_series, 470, 'bars')

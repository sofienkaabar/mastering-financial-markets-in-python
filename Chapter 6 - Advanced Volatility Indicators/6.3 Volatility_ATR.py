from master_library import import_data, indicator_plot, atr

# download the historical values
my_time_series = import_data(name='DIS')
# call the function(s)
my_time_series = atr(my_time_series)
# plot
indicator_plot(my_time_series, 
               indicator='volatility',
               window=500,
               plot_type='bars',
               barriers=False,
               indicator_label='ATR')
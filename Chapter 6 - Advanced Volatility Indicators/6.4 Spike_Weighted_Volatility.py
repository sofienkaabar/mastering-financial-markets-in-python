from master_library import import_data, indicator_plot, spike_weighted_volatility

# download the historical values
my_time_series = import_data(name='DIS')
# call the function(s)
my_time_series = spike_weighted_volatility(my_time_series)
# plot
indicator_plot(my_time_series, 
               indicator='swv',
               window=500,
               plot_type='bars',
               barriers=False,
               indicator_label='Spike Weighted Volatility')
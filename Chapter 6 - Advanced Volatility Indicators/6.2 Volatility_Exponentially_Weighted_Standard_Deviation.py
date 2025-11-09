from master_library import import_data, indicator_plot, exponentially_weighted_standard_deviation

# download the historical values
my_time_series = import_data(name='DIS')
# call the function(s)
my_time_series = exponentially_weighted_standard_deviation(my_time_series)
# plot
indicator_plot(my_time_series, 
               indicator='volatility',
               window=500,
               plot_type='bars',
               barriers=False,
               indicator_label='Exponentially-Weighted Standard Deviation')
from master_library import import_data, signal_chart_indicator, marsi

# download the historical values
my_time_series = import_data(name='DIS')
# call the function(s)
my_time_series = marsi(my_time_series, 20, 14)
# plot
signal_chart_indicator(my_time_series, 
                       'MARSI', 
                       window=800, 
                       lower_barrier=2, 
                       upper_barrier=98, 
                       plot_type='bars',
                       barriers=True,
                       indicator_label='MARSI')
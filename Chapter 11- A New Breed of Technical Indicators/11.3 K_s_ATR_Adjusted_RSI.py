from master_library import import_data, atr_adjusted_rsi, indicator_plot

# download the historical values
my_time_series = import_data('MSFT')
# call the function(s)
my_time_series = atr_adjusted_rsi(my_time_series, source='close')
# plot
indicator_plot(my_time_series, 'atr_adjusted_rsi', 
               window=500, 
               lower_barrier=35, 
               upper_barrier=70, 
               plot_type='bars',
               barriers=True,
               indicator_label='ATR Adjusted RSI')
from master_library import import_data, rsi, indicator_plot

# download the historical values
my_time_series = import_data(name='VIX')
my_time_series = rsi(my_time_series, source='close', output_name='RSI', rsi_lookback=5)
# plot distance correlation
indicator_plot(my_time_series, 'RSI', window=250, lower_barrier=20, upper_barrier=80, plot_type='bars', barriers=True,
               indicator_label='RSI')
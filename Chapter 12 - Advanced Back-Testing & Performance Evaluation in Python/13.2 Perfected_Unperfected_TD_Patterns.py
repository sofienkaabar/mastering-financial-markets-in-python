from master_library import import_data, td_setup, signal_chart, performance_evaluation

# download the historical values
my_time_series = import_data(name='NVDA')
# call the function(s)
my_time_series = td_setup(my_time_series, perfected=True)
# plot
signal_chart(my_time_series, window=500, choice='bars', source='open', chart_type='ohlc')
# performance evaluation
my_time_series = performance_evaluation(my_time_series, strategy='variable_holding_period', holding_period=10, theoretical_equity_curve=False)
from master_library import import_data, signal_chart, performance_evaluation, rob_booker_reversal

# download the historical values
my_time_series = import_data('AMZN')
# call the function(s)
my_time_series = rob_booker_reversal(my_time_series)
# plot
signal_chart(my_time_series, 1000, 'bars')
# performance evaluation
my_time_series = performance_evaluation(my_time_series, strategy='fixed_holding_period', holding_period=10, theoretical_equity_curve=True)
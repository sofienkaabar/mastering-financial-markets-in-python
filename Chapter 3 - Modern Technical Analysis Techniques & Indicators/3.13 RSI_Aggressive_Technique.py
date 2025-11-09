from master_library import import_data, rsi, signal_chart_indicator

def rsi_aggressive_technique(my_time_series, column='close', output_name='RSI', 
                                rsi_lookback=14, lower_barrier=30, upper_barrier=70):
    my_time_series = rsi(my_time_series, column, output_name, rsi_lookback=rsi_lookback)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['RSI'].iloc[i] < lower_barrier and \
           my_time_series['RSI'].iloc[i-1] > lower_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['RSI'].iloc[i] > upper_barrier and \
             my_time_series['RSI'].iloc[i-1] < upper_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
    
# download the historical values
my_time_series = import_data('JPM')
# call the function(s)
my_time_series = rsi_aggressive_technique(my_time_series)
# plot
signal_chart_indicator(my_time_series, indicator='RSI', window=250, lower_barrier=30, upper_barrier=70, plot_type='bars',
                       barriers=True, indicator_label='RSI')
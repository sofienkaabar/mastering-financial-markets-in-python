from master_library import import_data, candlestick_rsi, signal_chart_candlestick_rsi

def absolute_u_turn(my_time_series):
    my_time_series = candlestick_rsi(my_time_series, rsi_lookback=14)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):   
        try:
            # bullish signal
            if my_time_series['RSI_low'].iloc[i] > 20 and \
               my_time_series['RSI_low'].iloc[i-1] < 20 and \
               my_time_series['RSI_low'].iloc[i-2] < 20 and \
               my_time_series['RSI_low'].iloc[i-3] < 20 and \
               my_time_series['RSI_low'].iloc[i-4] < 20 and \
               my_time_series['RSI_low'].iloc[i-5] < 20:
                   my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
            # bearish signal
            elif my_time_series['RSI_high'].iloc[i] < 80 and \
                 my_time_series['RSI_high'].iloc[i-1] > 80 and \
                 my_time_series['RSI_high'].iloc[i-2] > 80 and \
                 my_time_series['RSI_high'].iloc[i-3] > 80 and \
                 my_time_series['RSI_high'].iloc[i-4] > 80 and \
                 my_time_series['RSI_high'].iloc[i-5] > 80:
                   my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1 
        except KeyError:
            pass
    return my_time_series 

# download the historical values
my_time_series = import_data(name='DIS')
# call the function(s)
my_time_series = absolute_u_turn(my_time_series)
# plot
signal_chart_candlestick_rsi(my_time_series, window=600, lower_barrier=20, upper_barrier=80, plot_type='bars', barriers=True)
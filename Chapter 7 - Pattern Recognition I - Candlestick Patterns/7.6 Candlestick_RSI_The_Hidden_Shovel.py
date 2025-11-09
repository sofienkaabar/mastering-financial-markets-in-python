from master_library import import_data, candlestick_rsi, signal_chart_candlestick_rsi

def hidden_shovel(my_time_series):
    my_time_series = candlestick_rsi(my_time_series, rsi_lookback=14)
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):   
        try:
            # bullish signal
            if my_time_series['RSI_low'].iloc[i] < 30 and \
               my_time_series['RSI_close'].iloc[i] > 30 and \
               my_time_series['RSI_open'].iloc[i] > 30 and \
               my_time_series['RSI_high'].iloc[i] > 30 and \
               my_time_series['RSI_low'].iloc[i-1] > 30:
                   my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
            # bearish signal
            elif my_time_series['RSI_high'].iloc[i] > 70 and \
                 my_time_series['RSI_close'].iloc[i] < 70 and \
                 my_time_series['RSI_open'].iloc[i] < 70 and \
                 my_time_series['RSI_low'].iloc[i] < 70 and \
                 my_time_series['RSI_high'].iloc[i-1] < 70:
                   my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1 
        except KeyError:
            pass
    return my_time_series 

# download the historical values
my_time_series = import_data(name='NKE')
# call the function(s)
my_time_series = hidden_shovel(my_time_series)
# plot
signal_chart_candlestick_rsi(my_time_series, window=500, lower_barrier=20, upper_barrier=80, plot_type='bars', barriers=True)
import matplotlib.pyplot as plt
from master_library import import_data, signal_chart_indicator, rsi, moving_average

def rsi_cross_technique(my_time_series, column='close', output_name='RSI', rsi_lookback=5, lower_barrier=25, upper_barrier=75):
    my_time_series = rsi(my_time_series, column, output_name, rsi_lookback=rsi_lookback)
    my_time_series = moving_average(my_time_series, source='RSI', ma_lookback=5, output_name='moving_average', ma_type='SMA')
    my_time_series['bullish_signal'] = 0
    my_time_series['bearish_signal'] = 0
    for i in range(0, len(my_time_series)):
        # bullish signal
        if my_time_series['RSI'].iloc[i] > my_time_series['moving_average'].iloc[i] and \
           my_time_series['RSI'].iloc[i-1] < my_time_series['moving_average'].iloc[i-1] and \
           my_time_series['RSI'].iloc[i] < lower_barrier:               
            my_time_series.at[my_time_series.index[i+1], 'bullish_signal'] = 1
        # bearish signal
        elif my_time_series['RSI'].iloc[i] < my_time_series['moving_average'].iloc[i] and \
             my_time_series['RSI'].iloc[i-1] > my_time_series['moving_average'].iloc[i-1] and \
             my_time_series['RSI'].iloc[i] > upper_barrier:
            my_time_series.at[my_time_series.index[i+1], 'bearish_signal'] = 1
    return my_time_series
    
# download the historical values
my_time_series = import_data('DIS')
# call the function(s)
my_time_series = rsi_cross_technique(my_time_series)
# plot
signal_chart_indicator(my_time_series, 
                       'RSI', 
                       window = 450, 
                       lower_barrier = 15, 
                       upper_barrier = 85, 
                       plot_type = 'bars',
                       barriers = True,
                       indicator_label = 'RSI')
plt.plot(my_time_series.iloc[-450:,]['moving_average'], label='Moving Average', color = 'black', linestyle = 'dashed')
plt.legend()
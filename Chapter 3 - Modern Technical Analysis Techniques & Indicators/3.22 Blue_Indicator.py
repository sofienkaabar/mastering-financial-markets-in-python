import matplotlib.pyplot as plt
from master_library import import_data, ohlc_plot, blue_indicator

def custom_signal_chart(my_time_series, window, choice='bars', source='open', chart_type='ohlc'): 
    sample = my_time_series.iloc[-window:, ]
    if chart_type == 'ohlc':
        ohlc_plot(sample, window, plot_type = choice)    
        for i in my_time_series.index:
            if my_time_series.loc[i, 'bullish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source] - 1),
                            arrowprops=dict(facecolor='blue', shrink=0.05))
            elif my_time_series.loc[i, 'bearish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source] + 1),
                            arrowprops=dict(facecolor='blue', shrink=0.05))    
    elif chart_type == 'simple':
        ohlc_plot(sample, window, plot_type = 'line', chart_type='simple')  
        for i in my_time_series.index:
            if my_time_series.loc[i, 'bullish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source] - 1),
                            arrowprops=dict(facecolor='blue', shrink=0.05))
            elif my_time_series.loc[i, 'bearish_signal'] == 1:
                plt.annotate('', xy=(i, my_time_series.loc[i, source]), xytext=(i, my_time_series.loc[i, source] + 1),
                            arrowprops=dict(facecolor='blue', shrink=0.05)) 
    plt.tight_layout()
    from matplotlib.lines import Line2D
    bullish_signal = Line2D([0], [0], marker='^', color='w', label='Buy signal', markerfacecolor='blue', markersize=10)
    bearish_signal = Line2D([0], [0], marker='v', color='w', label='Sell signal', markerfacecolor='blue', markersize=10)
    plt.legend(handles=[bullish_signal, bearish_signal]) 

# download the historical values
my_time_series = import_data('AAPL')
# call the function(s)
my_time_series = blue_indicator(my_time_series)
# plot
custom_signal_chart(my_time_series, 500)
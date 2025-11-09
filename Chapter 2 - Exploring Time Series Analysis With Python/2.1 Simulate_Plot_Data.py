from master_library import generate_ohlc_data, ohlc_plot

# generate ohlc data
my_time_series = generate_ohlc_data(length_data=250)
# plot bars
ohlc_plot(my_time_series, window=250, plot_type='bars')
# plot candlesticks
ohlc_plot(my_time_series, window=250, plot_type='candlesticks')
# plot line
ohlc_plot(my_time_series, window=250, plot_type='line')
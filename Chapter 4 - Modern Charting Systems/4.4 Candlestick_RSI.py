import matplotlib.pyplot as plt
from master_library import import_data, candlestick_rsi

def candlestick_rsi_plot(my_time_series, 
                         window=250, 
                         lower_barrier=20, 
                         upper_barrier=80, 
                         plot_type='bars',
                         barriers=True): 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # choose a sampling window 
    sample = my_time_series.iloc[-window:, ]   
    if plot_type == 'bars':
        for i in sample.index:  
            ax1.vlines(x=i, ymin=sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='black', linewidth=1)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='black', linewidth=1)  
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+1, color='black', linewidth=1)   
    elif plot_type == 'candlesticks':
        for i in sample.index:  
            ax1.vlines(x=i, ymin = sample.at[i, 'low'], ymax=sample.at[i, 'high'], color='black', linewidth=1)  
            if sample.at[i, 'close'] > sample.at[i, 'open']: 
                ax1.vlines(x=i, ymin=sample.at[i, 'open'], ymax=sample.at[i, 'close'], color='green', linewidth=3)  
            if sample.at[i, 'close'] < sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open'], color='red', linewidth=3)   
            if sample.at[i, 'close'] == sample.at[i, 'open']:
                ax1.vlines(x=i, ymin=sample.at[i, 'close'], ymax=sample.at[i, 'open']+3, color='black', linewidth=5)  
    else:
        print('Choose between bars or candlesticks')           
    for i in sample.index:  
        ax2.vlines(x=i, ymin=sample.at[i, 'RSI_low'], ymax=sample.at[i, 'RSI_high'], color='black', linewidth=1)  
        if sample.at[i, 'RSI_close'] > sample.at[i, 'RSI_open']: 
            ax2.vlines(x=i, ymin=sample.at[i, 'RSI_open'], ymax=sample.at[i, 'RSI_close'], color='green', linewidth=3)  
        if sample.at[i, 'RSI_close'] < sample.at[i, 'RSI_open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'RSI_close'], ymax=sample.at[i, 'RSI_open'], color='red', linewidth=3)   
        if sample.at[i, 'RSI_close'] == sample.at[i, 'RSI_open']:
            ax2.vlines(x=i, ymin=sample.at[i, 'RSI_close'], ymax=sample.at[i, 'RSI_open'], color='black', linewidth=2)
    if barriers == True:
        ax2.axhline(y=lower_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=upper_barrier, color='black', linestyle='dashed')
        ax2.axhline(y=50, color='black', linestyle='dashed')        
    ax1.grid()
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    
# download the historical values
my_time_series = import_data(name='AMZN')
# call the function(s)
my_time_series = candlestick_rsi(my_time_series, rsi_lookback=14)
# plot
candlestick_rsi_plot(my_time_series)

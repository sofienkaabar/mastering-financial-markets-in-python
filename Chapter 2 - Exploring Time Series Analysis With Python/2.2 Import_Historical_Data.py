from master_library import import_data

# import apple daily stock data from yahoo finance
my_time_series_yf = import_data(name='AAPL', start_date='2017-01-01', end_date='2025-06-01', data_provider='yahoo_finance', time_frame='daily')
# import eurusd hourly stock data from metatrader5
my_time_series_mt = import_data(name='EURUSD', start_date='2017-01-01', end_date='2025-06-01', data_provider='metatrader', time_frame='hourly')
# import consumer price index monthly data from fred
my_time_series_fr = import_data(name='CPIAUCSL', start_date='2017-01-01', end_date='2025-06-01', data_provider='fred')
# import tnote-10 etf data manually (xlsx)
my_time_series_xlsx = import_data(name='tnote_10_etf.xlsx', data_provider='manual_import_xlsx')
# import louis vuitton historical data manually (csv)
my_time_series_csv = import_data(name='louis_vuitton.csv', data_provider='manual_import_csv')
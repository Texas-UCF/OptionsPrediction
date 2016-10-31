from zipline.api import (
	# add_history,
	history,
	order_target_percent,
	order,
	record,
	symbol,
	get_datetime,
	schedule_function,
)
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo
import numpy as np 
import pandas as pd 
from datetime import datetime 

cash = 0 # tracks the amount of money in the backtest

def initialize(context):
	context.target_window = dict()
	context.bought_options = dict()

	# context.underlying = pd.read_csv('../data/underlying/FB.csv')
	# context.underlying = pd.to_datetime(context.underlying['Date'])

	context.options = pd.read_csv('../data/cleaned_data/FB.csv')
	context.options['date'] = pd.to_datetime(context.options['date'])
	context.options['expiration'] = pd.to_datetime(context.options['expiration'])


# (7) Trade (MODIFY SO THIS SHOULD ONLY HAPPEN ONCE A DAY)
def handle_data(context, data):
	day_option_df = context.options[context.options['date'] == get_datetime()]
	call_options = day_option_df[day_option_df['type'] == 'C']

	################################## classifier stuff happens somewhere here
	call_options_good = call_options # call_options_good is the classified call_options
	##################################

	# purchase the options that we think will end up in the money (could also modify this to give weight to it)
	for index, row in call_options_good.iterrows():
		context.bought_options = rbind(context.bought_options, row)
		cash -= row['price']

	# exercise expiring options that we've bought (assuming strike price is lower than expiration price)
	expiring_calls = context.bought_options[context.bought_options['expiration'] == get_datetime()]
	for index, row in expiring_calls.iterrows():
		price = history(symbol(row['ticker']), '1d', 'price').iloc[0,0]
        cash += 100*max(price - row['strike'], 0) # assuming 100:1 ratio equity:option

    # need to add a way to plot cash data vs datetime

def add_to_window(context, window_size, datapoint, ticker):
	tw = context.target_window[ticker]
	tw.append(datapoint)
	context.target_window[ticker] = tw[-window_size:] if len(tw) > window_size else tw

if __name__ == '__main__':
	cash = 10000 # arbitrary amount


	universe = ['FB'] # need to change the universe
	data = load_from_yahoo(stocks=universe,
		indexes={}, start=datetime(2016, 4, 3), 
		end=datetime.today())  
	


	olmar = TradingAlgorithm(initialize=initialize, handle_data=handle_data, capital_base=10000)  
	backtest = olmar.run(data)
	backtest.to_csv('backtest-50-2012.csv') 
	print backtest['algorithm_period_return'][-1]

	import pyfolio as pf
	returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(backtest)
	pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions, gross_lev=gross_lev, live_start_date='2004-10-22')
	
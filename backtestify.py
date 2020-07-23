import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn

try:
	from trade import Trade
except:
	pass
try:
	from backtest.trade import Trade
except:
	pass

import chart_studio.plotly as py	
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pd.options.display.float_format = '{:.5f}'.format
import random


class Backtest:  	
	def __init__(self, strategy, data, from_date, to_date, balance=10000, leverage=0, max_units=10000000, verbose=True, ipynb=False, direct=True, test=False, ddw=0, 
						commission=0.0, rfr=0.02):
		# initial variables
		self.strategy = strategy										# trading strategy
		self.Leverage = leverage										# leverage
		self.FromDate = str(from_date).split(' ')[0]					# starting date
		self.ToDate = str(to_date).split(' ')[0]                        # ending date
		self.Data = self.section(data, self.FromDate, self.ToDate)      # slice from the dataset
		self.Data['MC'] = ((self.Data['AC'] + self.Data['BC']) / 2)		# middle close price
		self.Data['MO'] = ((self.Data['AO'] + self.Data['BO']) / 2)		# middle open price
		self.Datasets = []												# all datasets nad instruments
		self.Verbose = verbose											# verbose checkker
		self.ipynb = ipynb												# only for Jupyter notebook
		self.Direct = direct											# calculating instrument units directly or indirectly
		self.Test = test												# run as a test only, with no balance calculation
		self.DDW = ddw													# drawdown value
		self.RfR = rfr													# risk-free rate
		
		# variables for the simulation
		self.Commission = commission									# commision per trade (percentage)
		self.OpenPositions = []											# list of the opened trades
		self.CurrentProfit = 0											# unrealized profit/loos
		self.GrossLoss = 0												# total loss
		self.GrossProfit = 0											# total profit
		self.TotalPL = 0												# total profit/loss
		self.InitBalance = balance										# initial balance
		self.Balance = balance											# account balance with closed trades
		self.MarginLeft	= balance										# margin left with unrealized profit	
		self.Unrealized = 0												# unrealized profit/loss
		self.MaxUnits = max_units										# maximal trading ammount
		self.History = []												# list to store previus prices for the user
		self.IndicatorList = []											# list to store indicators
		columns=['Type', 'Open Time', 'Close Time', 'Units', 'Margin Used', 'Open Price', 'Close Price', 'Spread', 'Profit',  'Balance', 'AutoClose', 'TP', 'SL']
		self.Results = pd.DataFrame(columns = ['Ratio', 'Value']) 		# dataframe for result analysis
		self.TradeLog = pd.DataFrame(columns = columns)					# pandas dataframe to log activity
		self.AutoCloseCount = 0											# counts how many times were trades closed automatically 
		snp_benchmark = None											# loading S&P as benchmark
		dji_benchmark = None											# loading DJI as benchmark
		dax_benchmark = None											# loading DAX as benchmark
		try:
			snp_benchmark = pd.read_csv('data/datasets/spx500usd/spx500usd_hour.csv')
		except:
			snp_benchmark = pd.read_csv('../data/datasets/spx500usd/spx500usd_hour.csv')

		try:
			dji_benchmark = pd.read_csv('data/datasets/djiusd/djiusd_hour.csv')
		except:
			dji_benchmark = pd.read_csv('../data/datasets/djiusd/djiusd_hour.csv')

		try:
			dax_benchmark = pd.read_csv('data/datasets/de30eur/de30eur_hour.csv')
		except:
			dax_benchmark = pd.read_csv('../data/datasets/de30eur/de30eur_hour.csv')

		self.DJI_Benchmark = self.section(dji_benchmark, self.FromDate, self.ToDate)
		self.SNP_Benchmark = self.section(snp_benchmark, self.FromDate, self.ToDate)
		self.DAX_Benchmark = self.section(dax_benchmark, self.FromDate, self.ToDate)


	def add_ma(self, n):
		name = 'MA' + str(n)
		self.IndicatorList.append(name)
		self.Data[name] = self.Data['MC'].rolling(n).mean()



	def add_wma(self, n):
		name = 'WMA' + str(n)
		self.IndicatorList.append(name)
		weights = np.arange(1,n+1)
		self.Data[name] = self.Data['MC'].rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)



	def add_ema(self, n):
		name = 'EMA' + str(n)
		self.IndicatorList.append(name)
		sma = self.Data['MC'].rolling(n).mean()
		mod_price = self.Data['MC'].copy()
		mod_price.iloc[0:10] = sma[0:10]
		self.Data[name] = mod_price.ewm(span=n, adjust=False).mean()



	def add_dema(self, n):
		name = 'DEMA' + str(n)
		self.IndicatorList.append(name)
		# calculating EMA
		sma = self.Data['MC'].rolling(n).mean()
		mod_price = self.Data['MC'].copy()
		mod_price.iloc[0:10] = sma[0:10]
		ema = mod_price.ewm(span=n, adjust=False).mean()
		# calculatung EMA of EMA
		sma_ema = ema.rolling(n).mean()
		mod_price_of_ema = ema.copy()
		mod_price_of_ema.iloc[0:10] = sma_ema[0:10]
		ema_of_ema = mod_price_of_ema.ewm(span=n, adjust=False).mean()
		self.Data[name] = 2 * ema - ema_of_ema



	def add_tema(self, n):
		name = 'TEMA' + str(n)
		self.IndicatorList.append(name)

		# calculating EMA
		sma = self.Data['MC'].rolling(n).mean()
		mod_price = self.Data['MC'].copy()
		mod_price.iloc[0:10] = sma[0:10]
		ema1 = mod_price.ewm(span=n, adjust=False).mean()
		
		# calculatung EMA of EMA1
		sma_ema1 = ema1.rolling(n).mean()
		mod_price_of_ema1 = ema1.copy()
		mod_price_of_ema1.iloc[0:10] = sma_ema1[0:10]
		ema2 = mod_price_of_ema1.ewm(span=n, adjust=False).mean()

		# calculatung EMA of EMA
		sma_ema2 = ema2.rolling(n).mean()
		mod_price_of_ema2 = ema2.copy()
		mod_price_of_ema2.iloc[0:10] = sma_ema2[0:10]
		ema3 = mod_price_of_ema2.ewm(span=n, adjust=False).mean()
		self.Data[name] = (3 * ema1) - (3 * ema2) + ema3



	def add_heikin_ashi(self):
		self.IndicatorList.append('HAC')
		self.IndicatorList.append('HAO')
		self.Data['HAH'] = self.Data.max(axis=1)
		self.Data['HAL'] = self.Data.drop(['ACh', 'BCh']).min(axis=1)
		self.Data['HAC'] = 0.25 * (self.Data['BO'] + self.Data['BH'] + self.Data['BL'] +self.Data['BC'])
		self.Data['HAO'] = 0.5 * (self.Data[1:]['BO'] + self.Data[1:]['BC'])



	def section(self, dt, from_date, to_date):
		start = dt.index[dt['Date'] == from_date].tolist()[0]
		end = dt.index[dt['Date'] == to_date].tolist()
		end = end[len(end) - 1]
		return dt[start:end].reset_index()



	def buy(self, row, instrument, trade_ammount, stop_loss=0, take_profit=0, units=0):
		if not self.Test:
			units = trade_ammount * self.Balance * self.Leverage
			units = units - units * self.Commission
		else:
			units = trade_ammount * units * self.Leverage

		if not self.Direct:
			units /= row['AC']
		if units > self.MaxUnits:
			units = self.MaxUnits
		
		self.OpenPositions.append(Trade(instrument[:6], 'BUY', units, row, stop_loss, take_profit, self.Direct))
		return True
	
	

	def sell(self, row, instrument, trade_ammount, stop_loss=0, take_profit=0, units=0):
		if not self.Test:
			units = trade_ammount * self.Balance * self.Leverage
			units = units - units * self.Commission
		else:
			units = trade_ammount * units * self.Leverage

		if not self.Direct:
			units /= row['BC']
		if units > self.MaxUnits:
			units = self.MaxUnits

		self.OpenPositions.append(Trade(instrument[:6], 'SELL', units, row, stop_loss, take_profit, self.Direct))
		return True
	
	

	def close(self, row, idx):
		if len(self.OpenPositions) == 0:
			return
		trade = self.OpenPositions.pop(idx)
		trade.close(row)
		if trade.Profit > 0:
			self.GrossProfit += trade.Profit
		else:
			self.GrossLoss += trade.Profit
		self.TotalPL += trade.Profit
		self.Balance += trade.Profit

		if not self.Direct:
			self.TradeLog.loc[len(self.TradeLog)] = [trade.Type, trade.OT, trade.CT, trade.Units, trade.Units / self.Leverage,
												 trade.OP, trade.CP, trade.CP - trade.OP, trade.Profit, self.Balance, trade.AutoClose, trade.TP, trade.SL]	
		else:
			self.TradeLog.loc[len(self.TradeLog)] = [trade.Type, trade.OT, trade.CT, trade.Units, trade.Units / self.Leverage,
												 trade.OP, trade.CP, trade.CP - trade.OP, trade.Profit, self.Balance, trade.AutoClose, trade.TP, trade.SL]



	def close_all(self, row):
		j = len(self.OpenPositions)
		while j != 0:
			self.close(row, 0)
			j -= 1

	
	def max_dd(self, data_slice):
		max2here = data_slice.expanding().max()
		dd2here = data_slice - max2here
		return dd2here.min()
	

	
	def run(self):
		simulation = None
		if self.Verbose:
			if not self.ipynb:
				simulation = tqdm(range(len(self.Data)))
			else:
				simulation = tqdmn(range(len(self.Data)))
		else:
			simulation = range(len(self.Data))	

		for i in simulation:
			if self.Verbose:
				simulation.set_description('Balance: {:.2f}'.format(self.Balance))
			row = self.Data.loc[i]
			self.Unrealized = 0

			for trade in self.OpenPositions:
				if not trade.Closed and (trade.update(row)):
					self.AutoCloseCount += 1
				else:
					self.Unrealized += trade.Profit
			
			j = 0
			while j < len(self.OpenPositions):
				if self.OpenPositions[j].Closed:
					self.close(row, j)
				j += 1
			
			if not self.Test:
				if self.Unrealized < -self.Balance:
					self.close_all(row)
					if self.Verbose:
						print('[INFO] Test stopped, inefficient funds.') 
					break

			self.strategy(self, row, i)
		self.close_all(row)


		# analysis
		if len(self.TradeLog) > 0:
			if self.DDW != 0:
				self.TradeLog['Drawdown'] = self.TradeLog['Balance'].rolling(self.DDW).apply(self.max_dd)
			else:
				dd_length = len(self.Data) / len(self.TradeLog)
				elf.TradeLog['Drawdown'] = self.TradeLog['Balance'].rolling(dd_length).apply(self.max_dd)

			columns = ['Nr. of Trades', 'Profit / Loss', 'Profit Factor', 'Win Ratio', 'Average P/L', 'Drawdown', 'DDW (%)', 'Buy & Hold', 'Sharpe Ratio', 'Balance', 'Max. Balance', 
			'Min. Balance', 'Gross Profit', 'Gross Loss', 'Winning Trades', 'Losing Trades', 'Average Profit', 'Average Loss', 'Profit Std.', 'Loss Std.', 'SL/TP Activated']

			if self.GrossLoss == 0:
				self.GrossLoss = 1

			buy = self.TradeLog[self.TradeLog['Type'] == 'BUY']
			buy_values = [len(buy), buy['Profit'].sum(), buy[buy['Profit'] > 0]['Profit'].sum() / abs(buy[buy['Profit'] < 0]['Profit'].sum()), 
					len(buy[buy['Profit'] > 0]) / len(buy), buy['Profit'].sum() / len(buy), None, None, None, None, None, None, None,
					buy[buy['Profit'] > 0]['Profit'].sum(), buy[buy['Profit'] < 0]['Profit'].sum(),
					len(buy[buy['Profit'] > 0]), len(buy[buy['Profit'] < 0]), 
					buy.loc[buy['Profit'] > 0]['Profit'].mean(), buy.loc[buy['Profit'] < 0]['Profit'].mean(), 
					buy.loc[buy['Profit'] > 0]['Profit'].std(), buy.loc[buy['Profit'] < 0]['Profit'].std(), 
					buy['AutoClose'].sum()]

			sell = self.TradeLog[self.TradeLog['Type'] == 'SELL']
			sell_values = [len(sell), sell['Profit'].sum(), sell[sell['Profit'] > 0]['Profit'].sum() / abs(sell[sell['Profit'] < 0]['Profit'].sum()), 
					len(sell[sell['Profit'] > 0]) / len(sell), sell['Profit'].sum() / len(sell), None, None, None, None, None, None, None,
					sell[sell['Profit'] > 0]['Profit'].sum(), abs(sell[sell['Profit'] < 0]['Profit'].sum()),
					len(sell[sell['Profit'] > 0]), len(sell[sell['Profit'] < 0]), 
					sell.loc[sell['Profit'] > 0]['Profit'].mean(), sell.loc[sell['Profit'] < 0]['Profit'].mean(), 
					sell.loc[sell['Profit'] > 0]['Profit'].std(), sell.loc[sell['Profit'] < 0]['Profit'].std(), 
					sell['AutoClose'].sum()]
			
			BnH = (self.Data['BC'][len(self.Data)-1] - self.Data['AC'][0]) * (1 / self.Data['BC'][len(self.Data)-1]) * 10000 * self.Leverage
			if not self.Direct:
				BnH = (self.Data['BC'][len(self.Data)-1] - self.Data['AC'][0]) * 10000 * self.Leverage / self.Data['AC'][0]

			sharpe_ratio = (self.Balance / self.InitBalance - 1 - self.RfR) / (self.TradeLog['Balance'] / self.InitBalance).std()
			all_values = [len(self.TradeLog), self.TotalPL, self.GrossProfit / abs(self.GrossLoss), len(self.TradeLog[self.TradeLog['Profit'] > 0]) / len(self.TradeLog), 
					self.TradeLog['Profit'].sum() / len(self.TradeLog), self.TradeLog['Drawdown'].min(), abs(self.TradeLog['Drawdown'].min()) / self.TradeLog['Balance'].max(),
					BnH, sharpe_ratio, self.Balance, self.TradeLog['Balance'].max(), self.TradeLog['Balance'].min(), self.GrossProfit, self.GrossLoss, 
					len(self.TradeLog[self.TradeLog['Profit'] > 0]), len(self.TradeLog[self.TradeLog['Profit'] < 0]), 
					self.TradeLog.loc[self.TradeLog['Profit'] > 0]['Profit'].mean(), self.TradeLog.loc[self.TradeLog['Profit'] < 0]['Profit'].mean(), 
					self.TradeLog.loc[self.TradeLog['Profit'] > 0]['Profit'].std(), self.TradeLog.loc[self.TradeLog['Profit'] < 0]['Profit'].std(), 
					self.AutoCloseCount]
			
			self.Results['Ratio'] = columns
			self.Results['All'] = all_values
			self.Results['Long'] = buy_values
			self.Results['Short'] = sell_values
	
	

	
	def plot_results(self, name='backtest_result.html'):    	
		if (len(self.TradeLog) > 0):
			fig = subplots.make_subplots(rows=3, cols=3, column_widths=[0.55, 0.27, 0.18],
									specs=[[{}, {}, {"rowspan": 2, "type": "table"}], 
											[{}, {}, None], 
											[{}, {"type": "table", "colspan": 2}, None]],
									shared_xaxes=True,
									subplot_titles=("Balance", "Benchmarks", "Performance Analysis", "Profit and Loss", "Monte Carlo Simulation", "Entries and Exits", "List of Trades"), 
									vertical_spacing=0.06, horizontal_spacing=0.02)

			buysell_color = []
			entry_shape = []
			profit_color = []
			for _, trade in self.TradeLog.iterrows():
				if trade['Type'] == 'BUY':
					buysell_color.append('#83ccdb')
					entry_shape.append('triangle-up')
				else:
					buysell_color.append('#ff0050')
					entry_shape.append('triangle-down')
				if trade['Profit'] > 0:
					profit_color.append('#cdeaf0')
				else:
					profit_color.append('#ffb1cc')


			buysell_marker = dict(color=buysell_color, size=self.TradeLog['Profit'].abs() / self.TradeLog['Profit'].abs().max() * 40)

			balance_plot = go.Scatter(x=pd.concat([pd.Series([self.TradeLog['Open Time'][0]]), self.TradeLog['Close Time']]), 
										y=pd.concat([pd.Series([self.InitBalance]), self.TradeLog['Balance']]), 
										name='Balance', connectgaps=True, fill='tozeroy', line_color="#5876F7")

			drawdown_plot = go.Scatter(x=pd.concat([pd.Series([self.TradeLog['Open Time'][0]]), self.TradeLog['Close Time']]), 
										y=pd.concat([pd.Series([0]), self.TradeLog['Drawdown']]), 
										name='DDW' + ' ' + str(self.DDW), connectgaps=True, fill='tozeroy', line_color="#ff0050")

			bubble_plot = go.Scatter(x=self.TradeLog['Close Time'], y=self.TradeLog['Profit'], name='P/L',
									marker=buysell_marker, mode='markers',
									hovertemplate = '<i>P/L</i>: %{y:.5f}' + '<b>%{text}</b>',
									text='<br>ID: ' + self.TradeLog.index.astype(str) + 
										 '<br>OP: ' + self.TradeLog['Open Price'].astype(str) +
										 '<br>CP: ' + self.TradeLog['Close Price'].astype(str) +
										 '<br>OT: ' + self.TradeLog['Open Time'] +
										 '<br>CT: ' + self.TradeLog['Close Time'] +
										 '<br>Units: ' + self.TradeLog['Units'].astype(str))

			profit_plot = go.Scatter(x=self.TradeLog['Close Time'], y=self.TradeLog['Profit'], name='Profit', 
									connectgaps=True, marker=dict(color='#1d3557'))

			price_plot = go.Scatter(x=self.Data['Date'] + ' ' + self.Data['Time'], y=self.Data['MC'], name='Price', 
									connectgaps=True, marker=dict(color='#b7c0fa'))

			entry_plot = go.Scatter(x=self.TradeLog['Open Time'], y=self.TradeLog['Open Price'], name='Entry', 
									marker=dict(color=buysell_color, symbol=entry_shape, size=14, opacity=0.7, line=dict(color='white', width=1)), mode='markers', hovertemplate = '<i>Price</i>: %{y:.5f}' + '<b>%{text}</b>',
									text='<br>ID: ' + self.TradeLog.index.astype(str) + 
										 '<br>Time: ' + self.TradeLog['Open Time'] + 
										 '<br>Units: ' + self.TradeLog['Units'].astype(str))
			
			exit_plot = go.Scatter(x=self.TradeLog['Close Time'], y=self.TradeLog['Close Price'], name='Exit', 
									marker=dict(color=buysell_color, size=10, opacity=0.7, line=dict(color='white', width=1)), mode='markers', 
									hovertemplate = '<i>Price</i>: %{y:.5f}' + '<b>%{text}</b>',
									text='<br>ID: ' + self.TradeLog.index.astype(str) + 
										 '<br>Time: ' + self.TradeLog['Close Time'] + 
										 '<br>Units: ' + self.TradeLog['Units'].astype(str) +
										 '<br>P/L: ' + self.TradeLog['Profit'].astype(str))
			
			not_needed = ['TP', 'SL', 'AutoClose', 'Drawdown']
			trade_list = go.Table(header=dict(values=self.TradeLog.drop(not_needed, axis=1).columns), 
					   			  cells=dict(values=[self.TradeLog.drop(not_needed, axis=1)[column] for column in self.TradeLog.drop(not_needed, axis=1)], 
											 format=[None] * 3 + [".2f"] * 2 + [".5f"] * 3 + [".2f"] * 2,
											 font=dict(color=['rgb(40, 40, 40)'] * 10, size=11),
											 fill_color=[profit_color * 10]))

			result_list = go.Table(header=dict(values=['Ratio', 'All', 'Long', 'Short']), 
								   cells=dict(values=[self.Results['Ratio'], self.Results['All'], self.Results['Long'], self.Results['Short']],
								   			  format=[None] + [".2f"],
											  height = 40,
											  fill = dict(color=['#C7D4E2', ' #EAF0F8'])))
			
			self.SNP_Benchmark = self.SNP_Benchmark.iloc[::int(len(self.SNP_Benchmark) / len(self.TradeLog)), :]
			snp_plot = go.Scatter(x=self.SNP_Benchmark['Date'] + ' ' + self.SNP_Benchmark['Time'], y=self.SNP_Benchmark['AC'] / self.SNP_Benchmark['AC'][0] - 1, name='S&P', 
									connectgaps=True, marker=dict(color='#b7c0fa'))

			self.DJI_Benchmark = self.DJI_Benchmark.iloc[::int(len(self.DJI_Benchmark) / len(self.TradeLog)), :]
			dji_plot = go.Scatter(x=self.DJI_Benchmark['Date'] + ' ' + self.DJI_Benchmark['Time'], y=self.DJI_Benchmark['AC'] / self.DJI_Benchmark['AC'][0] - 1, name='DJI', 
									connectgaps=True, marker=dict(color='#F35540'))

			self.DAX_Benchmark = self.DAX_Benchmark.iloc[::int(len(self.DAX_Benchmark) / len(self.TradeLog)), :]
			dax_plot = go.Scatter(x=self.DAX_Benchmark['Date'] + ' ' + self.DAX_Benchmark['Time'], y=self.DAX_Benchmark['AC'] / self.DAX_Benchmark['AC'][0] - 1, name='DAX', 
									connectgaps=True, marker=dict(color='#FECB52'))

			benchmark_plot = go.Scatter(x=pd.concat([pd.Series(self.Data['Date'][0] + ' ' + self.Data['Time'][0]), self.TradeLog['Close Time']]), 
										y=pd.concat([pd.Series([self.InitBalance]), self.TradeLog['Balance']]) / self.InitBalance - 1, name='Benchmark', 
										connectgaps=True, line_color="#5876F7")
			
			balance_reference_plot = go.Scatter(x=[x for x in range(len(self.TradeLog))], y=self.TradeLog['Balance'], name='MC Ref', line_color="#5876F7")
			
			# calculating monte carlo simulation
			last_balance = self.TradeLog['Balance'][len(self.TradeLog)-1]
			avg = (self.TradeLog['Profit'].sum() / len(self.TradeLog)) / self.InitBalance
			std_dev = self.TradeLog['Profit'].std() / self.InitBalance
			num_reps = int(len(self.TradeLog) / 2)
			num_simulations = 10
			avg_at = 2

			monte_carlos = []
			for x in range(num_simulations):
				price_series = [last_balance]
				price = last_balance * (1 + np.random.normal(0, std_dev))
				price_series.append(price)
				for y in range(num_reps):
					price = price_series[len(price_series)-1] * (1 + np.random.normal(0, std_dev))
					price_series.append(price)
				monte_carlos.append(np.array(price_series))

				if len(monte_carlos) >= avg_at:
					monte_carlos = np.array(monte_carlos)
					summed = sum(monte_carlos)
					monte_carlo = summed / len(monte_carlos)
					monte_carlo_plot = go.Scatter(x=[x+len(self.TradeLog)-1 for x in range(len(monte_carlo))], y=monte_carlo, name='MC', mode='lines')
					fig.append_trace(monte_carlo_plot, 2, 2)
					monte_carlos = []
			

			fig.append_trace(balance_plot, 1, 1)
			fig.append_trace(drawdown_plot, 1, 1)
			fig.append_trace(profit_plot, 2, 1)
			fig.append_trace(bubble_plot, 2, 1)
			fig.append_trace(price_plot, 3, 1)
			fig.append_trace(exit_plot, 3, 1)
			fig.append_trace(entry_plot, 3, 1)
			fig.append_trace(trade_list, 3, 2)
			fig.append_trace(snp_plot, 1, 2)
			fig.append_trace(dji_plot, 1, 2)
			fig.append_trace(benchmark_plot, 1, 2)
			fig.append_trace(dax_plot, 1, 2)
			
			fig.append_trace(balance_reference_plot, 2, 2)
			fig.append_trace(result_list, 1, 3)
			
			fig.update_layout(xaxis_rangeslider_visible=False, title=go.layout.Title(
							text = 'Backtest Results (' + self.FromDate[:4] + ' - ' + self.ToDate[:4] + ')', xref="paper"))
			
			if self.ipynb:
				iplot(fig)
			else:
				plot(fig, filename=name)
		else:
			print("No data to plot!")


	
	def plot_indicators(self, name='backtest_indicators.html'):
		if len(self.IndicatorList) > 0:
			fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.02)

			price_plot = go.Scatter(x=self.Data['Date'] + ' ' + self.Data['Time'], y=self.Data['MC'], name='Price', connectgaps=True, marker=dict(color='#b7c0fa'))
			fig.append_trace(price_plot, 1, 1)

			for ind in self.IndicatorList:
				plt = go.Scatter(x=self.Data['Date'] + ' ' + self.Data['Time'], y=self.Data[ind], name=ind, connectgaps=True)
				fig.append_trace(plt, 1, 1)
			
			fig.update_layout(xaxis_rangeslider_visible=False, title=go.layout.Title(text = self.FromDate[:4] + ' - ' + 
								self.ToDate[:4] + ' Indicators', xref="paper"))
			
			if self.ipynb:
				iplot(fig)
			else:
				plot(fig, filename=name)
		else:
			print("No indicators to plot!")



	def save_results(self, name='trades.csv'):
		self.TradeLog.to_csv("trades.csv", index=False)
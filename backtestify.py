import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from trade import Trade
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pd.options.display.float_format = '{:.5f}'.format
import random


class Backtest:  	
	def __init__(self, strategy, data, from_date, to_date, balance=10000, leverage=0, max_units=2000000, verbose=False, ipynb=False):
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
		# variables for the simulation
		self.OpenPositions = []											# list of the opened trades
		self.CurrentProfit = 0											# unrealized profit/loos
		self.GrossLoss = 0												# total loss
		self.GrossProfit = 0											# total profit
		self.TotalPL = 0												# total profit/loss
		self.Balance = balance											# account balance
		self.Unrealized = 0												# unrealized profit/loss
		self.MaxUnits = max_units										# maximal trading ammount
		self.History = []												# list to store previus prices for the user
		self.IndicatorList = []											# list to store indicators
		columns=['Type', 'Market', 'Open Time', 'Close Time', 'Units', 'Margin Used', 'Open Price', 'Close Price', 'Spread', 'Profit',  'Balance']
		self.TradeLog = pd.DataFrame(columns = columns)					# pandas dataframe to log activity


	def add_ma(self, n):
		name = 'MA' + str(n)
		self.IndicatorList.append(name)
		self.Data[name] = ((self.Data['AC'] + self.Data['BC']) / 2).rolling(n).mean()


	def add_wma(self, n):
		name = 'WMA' + str(n)
		self.IndicatorList.append(name)
		weights = np.arange(1,n+1)
		self.Data[name] = ((self.Data['AC'] + self.Data['BC']) / 2).rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


	def add_ema(self, n):
		name = 'EMA' + str(n)
		self.IndicatorList.append(name)
		sma = ((self.Data['AC'] + self.Data['BC']) / 2).rolling(n).mean()
		modPrice = ((self.Data['AC'] + self.Data['BC']) / 2)
		modPrice.iloc[0:10] = sma[0:10]
		self.Data[name] = modPrice.ewm(span=n, adjust=False).mean()


	def section(self, dt, from_date, to_date):
		start = dt.index[dt['Date'] == from_date].tolist()[0]
		end = dt.index[dt['Date'] == to_date].tolist()
		end = end[len(end) - 1]
		return dt[start:end].reset_index()


	def buy(self, row, instrument, trade_ammount, stop_loss=0, take_profit=0):
		units = trade_ammount * self.Balance * self.Leverage
		if units > self.MaxUnits:
			units = self.MaxUnits
		self.OpenPositions.append(Trade(instrument, 'BUY', units, row, stop_loss, take_profit))
	
	
	def sell(self, row, instrument, trade_ammount, stop_loss=0, take_profit=0):
		units = trade_ammount * self.Balance * self.Leverage
		if units > self.MaxUnits:
			units = self.MaxUnits
		self.OpenPositions.append(Trade(instrument, 'SELL', units, row, stop_loss, take_profit))
	
	
	def close(self, row, idx):
		trade = self.OpenPositions.pop(idx)
		trade.close(row)
		if trade.Profit > 0:
			self.GrossProfit += trade.Profit
		else:
			self.GrossLoss += trade.Profit
		self.TotalPL += trade.Profit
		self.Balance += trade.Profit
		self.TradeLog.loc[len(self.TradeLog)] = [trade.Type, trade.Instrument, trade.OT, trade.CT, trade.Units, trade.Units / self.Leverage,
												 trade.OP, trade.CP, trade.CP - trade.OP, trade.Profit, self.Balance]	

	def close_all(self, row):
		for i in range(len(self.OpenPositions)):
			self.close(row, i)


	def run(self):
		for i in tqdm(range(len(self.Data))): 
			row = self.Data.loc[i]
			self.Unrealized = 0
			for trade in self.OpenPositions:
				trade.update(row)
				self.Unrealized += trade.Profit
			
			for i in range(len(self.OpenPositions)):
				if self.OpenPositions[i].Closed:
					self.close(row, i)
				
			if self.Unrealized < -self.Balance:
				self.close_all(row)
				print('[INFO] Test stopped, inefficient funds.') 
				break

			self.strategy(self, row, i)

		self.close_all(self.Data.loc[len(self.Data)-1])

		print("Number of trades made: ", len(self.TradeLog))
		print("Total profit: ", self.GrossProfit)
		print("Total loss: ", self.GrossLoss)
		print("Max. balance: ", self.TradeLog['Balance'].max())
		print("Min. balance: ", self.TradeLog['Balance'].min())
		print("Balance: ", self.Balance)




	def plot(self, name='backtest_plot.html'):    	
		if (len(self.TradeLog) > 0):
			fig = subplots.make_subplots(rows=3, cols=2, 
										specs=[[{}, {"rowspan": 2, "is_3d": True, "type": "surface"}], [{}, None], [{}, {}]],
										shared_xaxes=True, 
										subplot_titles=("Balance","Risk / Confidence / Return", "Profit and Loss", "Entries and Exits", "Indicators"), 
										vertical_spacing=0.075, horizontal_spacing=0.075,)

			buysell_color = []
			entry_shape = []
			for _, trade in self.TradeLog.iterrows():
				if trade['Type'] == 'BUY':
					buysell_color.append('#83ccdb')
					entry_shape.append('triangle-up')
				else:
					buysell_color.append('#ff0050')
					entry_shape.append('triangle-down')

			buysell_marker = dict(color=buysell_color, size=self.TradeLog['Profit'].abs() / self.TradeLog['Profit'].abs().max() * 40)

			balance_plot = go.Scatter(x=self.TradeLog['Close Time'], y=self.TradeLog['Balance'], name='Balance', connectgaps=True)

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

			price_plot = go.Scatter(x=self.Data['Date'] + ' ' + self.Data['Time'], y=self.Data['MO'], name='Price', 
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
			
			x = np.linspace(-5, 80, 10)
			y = np.linspace(-5, 60, 10)
			xGrid, yGrid = np.meshgrid(y, x)
			z = [[random.randint(6, 16) for y in range(10)] for x in range(10)]
			corr_plot = go.Surface(x=x, y=y, z=z, showscale=False, name='Correlation',
								   hovertemplate='Return: %{z:.2f} <br>Risk: %{x:.2f} <br>Confidence: %{y:.2f}')
										 
			layout = go.Layout(
				xaxis=dict(domain=[0, 0.6]),
				xaxis2=dict(domain=[0.7, 1]),
				yaxis2=dict(anchor="x2")
			)
			fig.append_trace(balance_plot, 1, 1)
			fig.append_trace(profit_plot, 2, 1)
			fig.append_trace(bubble_plot, 2, 1)
			fig.append_trace(price_plot, 3, 1)
			fig.append_trace(exit_plot, 3, 1)
			fig.append_trace(entry_plot, 3, 1)
			fig.append_trace(corr_plot, 1, 2)
			fig.append_trace(price_plot, 3, 2)

			for ind in self.IndicatorList:
				plt = go.Scatter(x=self.Data['Date'] + ' ' + self.Data['Time'], y=self.Data[ind], name=ind, connectgaps=True)
				fig.append_trace(plt, 3, 2)
			fig.update_layout(xaxis_rangeslider_visible=False, title=go.layout.Title(text = self.TradeLog['Close Time'][0][:4] + ' - ' + 
							  self.TradeLog['Close Time'][len(self.TradeLog) - 1][:4], xref="paper"))
						
			if self.ipynb:
				iplot(fig)
			else:
				plot(fig, filename=name)
		else:
			print("No data to plot!")

	def save_results(self, name='trades.csv'):
		self.TradeLog.to_csv("trades.csv", index=False)

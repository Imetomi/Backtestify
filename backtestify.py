import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pd.options.display.float_format = '{:.5f}'.format

ipynb = False
def notebook_mode():
	init_notebook_mode(True)
	ipynb = True

class Backtest:  
	
	def __init__(self, strategy, data, from_date, to_date, 
				balance=10000, leverage=30, resolution='Hour', 
				stop_loss_active=True, stop_loss=25,
				take_profit_active=True, take_profit=50, 
				max_units=2000000,
				verbose=False):
		
		# initial variables
		self.strategy = strategy
		self.leverage = leverage
		self.resolution = resolution
		self.stop_loss = stop_loss
		self.stop_loss_active = stop_loss_active
		self.from_date = str(from_date).split(' ')[0]
		self.to_date = str(to_date).split(' ')[0]
		self.data = self.prep_data(data, self.from_date, self.to_date)
		self.verbose = verbose

		# intializing for simulation
		self.current_trade = [0, None, 0, 0, None]
		self.ongoing_trade = False
		self.current_profit = 0
		self.gross_loss = 0
		self.gross_profit = 0
		self.total_profit_loss = 0
		self.current_pip = 0
		self.balance = balance
		self.max_units = max_units
		self.take_profit_active = take_profit_active
		self.take_profit = take_profit
		self.hist = []

		columns=['Type', 'Open Date', 'Close Date', 'Units', 'Margin Used', 'Start Price', 
				 'Close Price', 'PIPs', 'Profit', 'Balance']
		self.trade_list = pd.DataFrame(columns = columns)

		
	def prep_data(self, dt, from_date, to_date):
		start = dt.index[dt['Date'] == from_date].tolist()[0]
		end = dt.index[dt['Date'] == to_date].tolist()
		end = end[len(end) - 1]
		return dt[start:end].reset_index()


	def buy(self, row, trade_ammount=0.8):
		if self.ongoing_trade:
			return False
		else:
			margin = trade_ammount * self.balance
			units = margin * self.leverage
			if units > self.max_units:
				units = self.max_units
			self.current_trade = [row['AskOpen'], 'BUY', units, margin, row['Date'] + ' ' + str(row[self.resolution])]
			self.ongoing_trade = True
			return True
	
	
	
	def sell(self, row, trade_ammount=0.8):
		if self.ongoing_trade:
			 return False
		else:
			margin = trade_ammount * self.balance
			units = margin * self.leverage
			if units > self.max_units:
				units = self.max_units
			self.current_trade = [row['BidOpen'], 'SELL', units, margin, row['Date'] + ' ' + str(row[self.resolution])]
			self.ongoing_trade = True
			return True
	
	
	
	def close(self, row):
		if self.ongoing_trade and self.current_trade[0] != 0:
			pass
		else: 
			return False
		
		self.balance += self.current_profit
		if self.current_trade[1] == 'BUY':
			self.trade_list.loc[len(self.trade_list)] = [self.current_trade[1], self.current_trade[4], 
							row['Date']+ ' ' + str(row[self.resolution]), self.current_trade[2], 
							self.current_trade[3], self.current_trade[0], row['BidOpen'], 
							self.current_profit / ((1 / row['BidOpen']) * (self.current_trade[2] / 10000)),
							self.current_profit, self.balance]
			
		else:
			self.trade_list.loc[len(self.trade_list)] = [self.current_trade[1], self.current_trade[4], 
							row['Date']+ ' ' + str(row[self.resolution]), self.current_trade[2], 
							self.current_trade[3], self.current_trade[0], row['AskOpen'],
							self.current_profit / ((1 / row['AskOpen']) * (self.current_trade[2] / 10000)),
							self.current_profit, self.balance]
	
		self.ongoing_trade = False
		
		if self.current_profit > 0:
			self.gross_profit += self.current_profit
		else:
			self.gross_loss += self.current_profit
		
		self.total_profit_loss += self.current_profit
		self.current_trade = [0, None, 0, 0, None]
		self.current_profit = 0
		return True
	
	
	def calc_profit(self, row):
		if self.current_trade[1] == 'BUY':
			return (row['BidOpen'] - self.current_trade[0]) * (1 / row['BidOpen']) * self.current_trade[2]
		elif self.current_trade[1] == 'SELL':
			return (self.current_trade[0] - row['AskOpen']) * (1 / row['AskOpen']) * self.current_trade[2]
		else:
			return 0
   
	def calc_pip(self, row):
		if self.current_trade[1] == 'BUY':
			return self.current_profit / ((1 / row['BidOpen']) * (self.current_trade[2] / 10000))
		elif self.current_trade[1] == 'SELL':
			return self.current_profit / ((1 / row['AskOpen']) * (self.current_trade[2] / 10000))
		else:
			return 0
	
	
	def run(self):
		for i in tqdm(range(len(self.data))): 
			row = self.data.loc[i]
			self.current_profit = self.calc_profit(row)
			self.current_pip = self.calc_pip(row)

			if self.take_profit_active:
				if self.current_pip >= self.take_profit:
					#if self.verbose:
					#	print('Take Profit Actvivated at ' + str(self.current_pip))
					self.close(row)
			if self.stop_loss_active:
				if self.current_pip <= -self.stop_loss:
					#if self.verbose:
					#	print('Stop Loss Actvivated at ' + str(self.current_pip))
					self.close(row)
			if self.current_profit <= -self.balance:
					self.close(row)
					return

			self.strategy(self, row)
			
			
			if i == (len(self.data) - 1):
				self.close(row)

		if self.verbose:
			print("Total number of trades made: ", len(self.trade_list))
			print("Total profit: ", self.gross_profit)
			print("Total loss: ", self.gross_loss)
			print("Max. balance: ", self.trade_list['Balance'].max())
			print("Balance: ", self.balance)




	def plot(self, name='backtest_plot.html'):    	
		if (len(self.trade_list) > 1):
			fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

			balance_plot = go.Scatter(x=self.trade_list['Close Date'], y=self.trade_list['Balance'], name='Balance')
			profit_plot = go.Scatter(x=self.trade_list['Close Date'], y=self.trade_list['Profit'], name='Profit')

			fig.append_trace(balance_plot, 1, 1)
			fig.append_trace(profit_plot, 2, 1)

			fig.update_xaxes(title_text="Time", row=1, col=1)
			fig.update_xaxes(title_text="Time", row=2, col=1)
			fig.update_yaxes(title_text="Balance", row=1, col=1)
			fig.update_yaxes(title_text="Profit", row=2, col=1)
			fig.update_layout(
					title=go.layout.Title(
						text = self.trade_list['Close Date'][0][:4] + ' - ' + self.trade_list['Close Date'][len(self.trade_list) - 1][:4],
						xref="paper",
						x=0
					)
			)
			
			if notebook_mode:
				iplot(fig)
			else:
				plot(fig, filename=name)
		else:
			print("No data to plot!")
			

from backtestify import Backtest
from datetime import datetime
import pandas as pd
import random

instrument = 'eurusd'
direct = True
data = pd.read_csv('data/' + instrument + '.csv')

fast = 'MA21'
slow = 'EMA50'
leverage = 30
SL = 1
TP = 1
starting_date = datetime(2019, 1, 4)
ending_date = datetime(2019, 7, 2)


def strategy(self, row, i):
	if self.Data[fast][i] > self.Data[slow][i] and self.Data[fast][i-1] < self.Data[slow][i-1]:
		self.close_all(row)
		self.buy(row, instrument, 1, SL * self.Balance, TP * self.Balance)
	elif self.Data[fast][i] < self.Data[slow][i] and self.Data[fast][i-1] > self.Data[slow][i-1]:
		self.close_all(row)
		self.sell(row, instrument, 1,  SL * self.Balance, TP * self.Balance)


backtest = Backtest(strategy, data, starting_date, ending_date, leverage=leverage, balance=10000, direct=direct, ddw=10)
backtest.add_ma(21)
backtest.add_ema(50)
backtest.run()
backtest.plot_results()
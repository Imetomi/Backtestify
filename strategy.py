from backtestify import Backtest
from datetime import datetime
import pandas as pd
import random


ins = 'eurusd'
data = pd.read_csv('data/eurusd.csv')


def strategy(self, row, i):
    if self.Data['EMA21'][i] > self.Data['EMA50'][i] and self.Data['EMA21'][i-1] < self.Data['EMA50'][i-1]:
        self.close_all(row)
        self.buy(row, ins, 0.8, self.Balance * 0.03, self.Balance * 0.01)
    elif self.Data['EMA21'][i] < self.Data['EMA50'][i] and self.Data['EMA21'][i-1] > self.Data['EMA50'][i-1]:
        self.close_all(row)
        self.sell(row, ins, 0.8, self.Balance * 0.03, self.Balance * 0.01)


backtest = Backtest(strategy, data, datetime(2018, 1, 2), datetime(2019, 1, 2), leverage=50, balance=1000)
backtest.add_ema(21)
backtest.add_ema(50)
backtest.save_results()

backtest.run()
backtest.plot()

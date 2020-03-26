from backtestify import Backtest
from datetime import datetime
import pandas as pd
import random

def strategy(self, row):
    if not self.ongoing_trade and len(self.hist) == 24:
        # Check if previous day's price was higher or lower.
        # If it's lower it means that today the price is low so we buy.
        
        if self.hist[23] - self.hist[0] > 0:
            self.buy(row)
        else:
            self.sell(row)

        self.hist = []
    else:
        self.hist.append(row['AskClose'])      

    if self.ongoing_trade:
        if row['index'] % 24 == 0:
            self.close(row)


data = pd.read_csv('data/eurusd.csv')
backtest = Backtest(strategy, data, datetime(2017, 1, 2), datetime(2019, 1, 14), verbose=True,
                    stop_loss=10, take_profit=10)
backtest.run()
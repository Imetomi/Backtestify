from backtestify import Backtest
from datetime import datetime
import pandas as pd
import random

def strategy(self, row):
    if not self.ongoing_trade and len(self.hist) == 24:
        
        # very simple "strategy", we buy and sell based on what happened a day before. 
        if random.randint(0, 1):
            # buy on the actual price
            self.buy(row)
        else:
            # sell on the actual price
            self.sell(row)

        # empty the list for the next 24 hours
        self.hist = []
    else:
        # a variable to store records for our strategy
        self.hist.append(row['AskOpen'])      

    # after a day close the open positions
    if self.ongoing_trade:
        if row['index'] % 24 == 0:
            # close the trade
            self.close(row)

# read the date (1H resolution)
data = pd.read_csv('data/eurusd.csv')

# initialize backtest
backtest = Backtest(strategy, data, datetime(2016, 1, 4), datetime(2019, 1, 2), verbose=True,
                    stop_loss=10, # stop losses at 10 pip
                    take_profit=10 # take profit at 10 pip
                    )

# run and test the algorithm
backtest.run()

# optionally save the trades in a CSV file 
backtest.trade_list.to_csv("trades.csv")

# create a plot of the account balance
backtest.plot()

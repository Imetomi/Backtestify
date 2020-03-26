# Backtestify

A simple backtest library to simulate real world trading. 

## About this Project

**Backtestify** takes spread into account while simulating your strategy. Most of the backtesting libraries out there tend to generate false profit caused by the lack of spread when placing and closing a market order. **Backtestify** uses real historical data from **Oanda.com** with correct difference between ask and bid price.
The drawback of the accurate trading simulation is that currently you can only have one open position which makes sense when you only use one instrument. I am building this library alone so development takes time.



## Implemented Features 

1. Placing long and short postiions in a simple way.
2. Take profit and stop loss.
3. Leverage
4. Plotting strategy results and account balance history.
5. Buy on ASK price, sell on BID price to avoid accidental fake profit.

![Backtest Result](https://github.com/Imetomi/Backtestify/blob/master/data/plot.png)

## Example code

Check out 'strategy.py'.


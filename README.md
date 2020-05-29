# Backtestify

A simple backtest library to simulate real world trading. 

## About this Project

**Backtestify** takes spread into account while simulating your strategy. Most of the backtesting libraries out there tend to generate false profit caused by the lack of spread when placing and closing a position. **Backtestify** uses real historical data from **Oanda.com** with correct difference between the ask and bid price.
The drawback of the accurate trading simulation is that currently you can have only one open position but this makes sense when you only use one instrument. 
I am building this library alone so development takes time.

## Implemented Features 

1. Placing long and short positions in a simple way.
2. Take profit and stop loss.
3. Leverage.
4. Plotting strategy results and account balance history.
5. Buy on ASK price, sell on BID price to avoid accidental fake profit.

![Backtest Result](https://github.com/Imetomi/Backtestify/blob/master/data/plot.PNG)

## Example code

Check out **strategy.py**.

## TODOs

Right now I'm working on scraping more real data from the web. The program is optimized for EUR/USD forex trading which means that PIP calculation may run into problems with other instruments.
 

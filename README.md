# Backtestify

A simple backtest library to simulate real world trading. 

## About this Project

**Backtestify** takes spread into account while simulating your strategy. Most of the backtesting libraries out there tend to generate false profit caused by the lack of spread when placing and closing a position. **Backtestify** uses real historical data from **Oanda.com** with correct difference between the ask and bid price.
I am building this library alone so development takes time.

## Implemented Features 

1. Placing long and short positions in a simple way.
2. Take profit and stop loss.
3. Leverage.
4. Plotting strategy results and account balance history.
5. Buy on ASK price, sell on BID price to avoid accidental fake profit.
6. Monte Carlo simulations based on your strategy's performance.
7. Benchmarking your strategy with SPY, DAX, and DJA.

![Backtest Result](https://github.com/Imetomi/Backtestify/blob/master/data/plot.png)

## Example code

Check out **strategy.py**.

 

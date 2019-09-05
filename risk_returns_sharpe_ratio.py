"""
Python project to analyze Sharpe Ratio of different stocks.

IPython Notebook is available under risk_returns_sharpe_ratio.ipynb
"""

# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading in the data
stock_data = pd.read_csv('datasets/stock_data.csv', parse_dates=['Date'], index_col='Date').dropna()
benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna()

# Display summary for stock_data (Uncomment for information)
# print('Stocks\n')
# print(stock_data.info())

# Display summary for benchmark_data (Uncomment for information)
# print('\nBenchmarks\n')
# print(benchmark_data.info())
'''
=OUTPUT=
Stocks

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 252 entries, 2016-01-04 to 2016-12-30
Data columns (total 2 columns):
Amazon      252 non-null float64
Facebook    252 non-null float64
dtypes: float64(2)
memory usage: 5.9 KB
None

Benchmarks

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 252 entries, 2016-01-04 to 2016-12-30
Data columns (total 1 columns):
S&P 500    252 non-null float64
dtypes: float64(1)
memory usage: 3.9 KB
None
'''

# visualize the stock_data (Uncomment for information)
# stock_data.plot(subplots=True, title='Stock Data')
# plt.show()

# summarize the stock_data (Uncomment for information)
# print(stock_data.describe())
'''
=OUTPUT=
           Amazon    Facebook
count  252.000000  252.000000
mean   699.523135  117.035873
std     92.362312    8.899858
min    482.070007   94.160004
25%    606.929993  112.202499
50%    727.875000  117.765000
75%    767.882492  123.902503
max    844.359985  133.279999
'''

# plot the benchmark_data (Uncomment for information)
# benchmark_data.plot()
# plt.show()

# summarize the benchmark_data (Uncomment for information)
# print(benchmark_data.describe())
'''
=OUTPUT=
           S&P 500
count   252.000000
mean   2094.651310
std     101.427615
min    1829.080000
25%    2047.060000
50%    2104.105000
75%    2169.075000
max    2271.720000
'''

# calculate daily stock_data returns (Uncomment for information)
stock_returns = stock_data.pct_change()

# plot the daily returns (Uncomment for information)
# stock_returns.plot()
# plt.show()

# summarize the daily returns (Uncomment for information)
# print(stock_returns.describe())
'''
=OUTPUT=
           Amazon    Facebook
count  251.000000  251.000000
mean     0.000818    0.000626
std      0.018383    0.017840
min     -0.076100   -0.058105
25%     -0.007211   -0.007220
50%      0.000857    0.000879
75%      0.009224    0.008108
max      0.095664    0.155214
'''

# calculate daily benchmark_data returns
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns (Uncomment for information)
# sp_returns.plot()
# plt.show()

# summarize the daily returns (Uncomment for information)
# print(sp_returns.describe())
'''
=OUTPUT=
count    251.000000
mean       0.000458
std        0.008205
min       -0.035920
25%       -0.002949
50%        0.000205
75%        0.004497
max        0.024760
Name: S&P 500, dtype: float64
'''

# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns (Uncomment for information)
# excess_returns.plot()
# plt.show()

# summarize the excess_returns (Uncomment for information)
# print(excess_returns.describe())
'''
=OUTPUT=
           Amazon    Facebook
count  251.000000  251.000000
mean     0.000360    0.000168
std      0.016126    0.015439
min     -0.100860   -0.051958
25%     -0.006229   -0.005663
50%      0.000698   -0.000454
75%      0.007351    0.005814
max      0.100728    0.149686
'''

# calculate the mean of excess_returns
avg_excess_return = excess_returns.mean()

# plot avg_excess_returns (Uncomment for information)
# avg_excess_return.plot.bar(title='Mean of the Return Difference')
# plt.show()

# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations (Uncomment for information)
# sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')
# plt.show()

# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio (Highest sharpe ratio is better)
annual_sharpe_ratio.plot.bar()
plt.show()
# Choose Amazon

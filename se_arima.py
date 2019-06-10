# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt     
plt.rcParams['font.sans-serif'] = ['SimHei']    #定义使其正常显示中文字体黑体
plt.rcParams['axes.unicode_minus'] = False      #用来正常显示表示负号
import warnings
warnings.filterwarnings("ignore")


data = pd.read_excel('car.xlsx', index_col = u'日期',header = 0)#brand_dazong.xlsx
print(data)

#画出时序图
#data.plot()
#plt.show()

#画出自相关性图
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#plot_acf(data)
#plt.show()

#平稳性检测
from statsmodels.tsa.stattools import adfuller
#返回值依次为：adf, pvalue p值， usedlag, nobs, critical values临界值 , 
# icbest, regresults, resstore 
#adf 分别大于3中不同检验水平的3个临界值，单位检测统计量对应的p 值显著大于 0.05 ， 
#说明序列可以判定为 非平稳序列
print('原始序列的检验结果为：',adfuller(data[u'销量']))

#对数据进行差分后得到 自相关图和 偏相关图
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']

#D_data.plot()   #画出差分后的时序图
#plt.show()
#plot_acf(D_data)    #画出自相关图
#plt.show()
#plot_pacf(D_data)   #画出偏相关图
#plt.show()
#一阶差分后的序列的时序图在均值附近比较平稳的波动， 自相关性有很强的短期相关性， 
#单位根检验 p值小于 0.05 ，所以说一阶差分后的序列是平稳序列
print(u'差分序列的ADF 检验结果为： ', adfuller(D_data[u'销量差分']))   #平稳性检验

#对一阶差分后的序列做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果：',acorr_ljungbox(D_data, lags= 1)) #返回统计量和 p 值
# 差分序列的白噪声检验结果： (array([*]), array([*])) p值为第二项， 远小于 0.05

#对模型进行定阶
from statsmodels.tsa.arima_model import ARIMA 
import itertools
import statsmodels.api as sm
# # Define the p, d and q parameters to take any value between 0 and 2
# p = d = q = range(0, 2)
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
# warnings.filterwarnings("ignore") # specify to ignore warning messages
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(D_data,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

mod = sm.tsa.statespace.SARIMAX(data,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 4),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
# results.plot_diagnostics(figsize=(15, 12))
# plt.show()

pred = results.get_prediction(start=70,dynamic=False)
pred_ci = pred.conf_int()
ax = data['1998':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('销量')
plt.legend()
plt.show()


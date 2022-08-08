from importlib.metadata import files
from symbol import parameters
import pandas as pd
import pylab
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import product
from scipy import stats
import numpy
import warnings



def invboxcox(y, lmbda):
    if lmbda == 0:
        return (pd.np.exp(y))
    else:
        return (pd.np.exp(pd.np.log(lmbda * y + 1) / lmbda))


a = pd.read_csv('pay22.csv', ';', index_col=['month'], parse_dates=['month'], dayfirst=True)
# plt.figure(figsize(15,7))
print(sm.tsa.stattools.adfuller(a))  # критерий дикки фуллера на стационарность
# sm.tsa.seasonal_decompose(a).plot()  # таблица
a.plot()
plt.show()
a["salary_box"], lmbda = stats.boxcox(a.salary)
# while sm.tsa.stattools.adfuller(a.salary_box)[1]>0.5:
#     a["salary_box"],lmbda=stats.boxcox(a.salary_box)
print("Оптимальный параметр преобр. Бокса-кокса:",lmbda)
print("Критерий Дики-Фуллера:",sm.tsa.stattools.adfuller(a.salary_box)[1])

a["salary_box_diff"] = a.salary_box - a.salary_box.shift(12)
a["salary_box_diff2"] = a.salary_box_diff - a.salary_box_diff.shift(1)
sm.tsa.seasonal_decompose(a.salary_box_diff2[13:]).plot()
plt.show() # Ряд вроде стал стационарным
# ax=plt.subplot(211)
# sm.graphics.tsa.plot_acf(a.salary_box_diff2[13:].values.squeeze(),lags=48,ax=ax)
# pylab.show()
# ax=plt.subplot(212)
# sm.graphics.tsa.plot_pacf(a.salary_box_diff2[13:].values.squeeze(),lags=48,ax=ax)
# pylab.show()


ps = range(0, 4)
Ps = range(0, 3)
qs = range(0, 2)
Qs = range(0, 2)
D = d = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(len(parameters_list))

results = []
best_aic = float("inf")
warnings.filterwarnings("ignore")
for param in parameters_list:
    try:
        model = sm.tsa.statespace.SARIMAX(a.salary_box, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except:
        print("wrong parameters:", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
        results.append([param, model.aic])
warnings.filterwarnings("default")

result_table = pd.DataFrame(results)
result_table.columns = ["parameters", "aic"]
print(result_table.sort_values(by="aic", ascending=[True]).head())
print(best_model.summary())

plt.subplot(211)
best_model.resid[13:].plot()
plt.show()
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)
pylab.show()
print("Критерий Стьюдента:", stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Критерий Дики-Фуллера:", sm.tsa.stattools.adfuller(best_model.resid[13:])[1])
a["model"] = invboxcox(best_model.fittedvalues, lmbda)
a.salary.plot()
a.model[13:].plot(color="r")
pylab.show()

# Прогноз
a2 = a[["salary"]]
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
date_list = [pylab.datetime.datetime.strptime("2021-11-01", "%Y-%m-%d") + pylab.relativedelta(months=x) for x in
             range(0, 49)]
future = pd.DataFrame(index=date_list, columns=a2.columns)
a2 = pd.concat([a2, future])
a2["forecast"] = invboxcox(best_model.predict(start=337, end=390), lmbda)
a2.salary.plot()
a2.forecast.plot(color="r")
pylab.show()
a2.to_csv("forecast22.csv")
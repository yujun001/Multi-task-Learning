import numpy as np
b = np.load('LSTM.npz')['arr_1']
# ['星期几：1-7' '小时：0-23' '白天温度' '夜间温度' '天气' 'aqi系数' 'aqi level']
print(b)
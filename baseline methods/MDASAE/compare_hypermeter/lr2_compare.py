import matplotlib.pyplot as plt
import numpy as np

x_axis_data = [0.0001, 0.0005, 0.001,0.01,0.1]  # x
y_axis_data = [0.9485,0.9608, 0.9550,0.9670, 0.9645]  # y
values = range(len(x_axis_data))
plt.plot(values, y_axis_data, 'b*--',label='auc')  # 'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
#plt.xscale('logit')
plt.xticks(values,x_axis_data)
plt.legend()  # 显示上面的label
plt.xlabel('head')  # x_label
plt.ylabel('AUC value')  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
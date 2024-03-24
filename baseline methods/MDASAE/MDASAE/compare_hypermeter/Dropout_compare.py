import matplotlib.pyplot as plt
import numpy as np

x_axis_data = [0.2, 0.3, 0.4,0.5,0.6]  # x
y_axis_data = [0.9492,0.949, 0.9493,0.9495, 0.9494]  # y

y = plt.FormatStrFormatter('%0.02f')
x_major_locator = plt.MultipleLocator(0.1)
#y_major_locator = plt.MultipleLocator(0.0001)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
#ax.yaxis.set_major_locator(y_major_locator)
ax.set_ylim(0.948,0.95)
ax.yaxis.set_minor_formatter(y)


plt.plot(x_axis_data, y_axis_data, 'b*--',label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

plt.legend()  # 显示上面的label
plt.xlabel('Dropout')  # x_label
plt.ylabel('AUC value')  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(4,4))
labels = [2, 4, 8,16]  # x
y = [0.9681,0.9690, 0.9665,0.9670]  # y
values = range(len(y))

plt.bar( values,y,tick_label=labels,width=0.4,color = ['xkcd:sky blue'])  # 'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
#plt.xscale('logit')
plt.tick_params(bottom=False, top=False, left=False, right=False)
#ax.tick_params(bottom=False, top=False, left=False, right=False)

plt.grid(axis='y')
ax = plt.gca()
ax.set_ylim(0.960,0.970)

#plt.xticks(values,x_axis_data)
#plt.legend()  # 显示上面的label
plt.xlabel('head')  # x_label
plt.ylabel('AUC value')  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
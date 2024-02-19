import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#设置坐标轴间隔
#准备数据

plt.figure(figsize=(5, 5))
data = [[0.9540, 0.9471],
[0.9690, 0.9663]]
X = np.arange(2)
year = ['5-flod','10-flod']
# 此处的 _ 下划线表示将循环取到的值放弃，只得到[0,1,2,3,4]
ind = [x for x, _ in enumerate(year)]


x= np.arange(len(year))
width = 0.1
#绘制柱状图
plt.bar(X, data[0],color = 'xkcd:sky blue', width = 0.2,label='MDASAE W/O attention')
plt.bar(X+0.2 , data[1], color = 'xkcd:peach', width = 0.2,label='MDASAE ')

W = [0.00,0.2]# 偏移量
for i in range(2):
    for a,b in zip(X+W[i],data[i]):#zip拆包
        plt.text(a,b,"%0.4f"% b,ha="center",va= "bottom")#格式化字符串，保留0位小数

plt.tick_params(bottom=False, top=False, left=False, right=False)

#ax.tick_params(bottom=False, top=False, left=False, right=False)

plt.grid(axis='y')
plt.ylim(0.9,1)

plt.xticks(x+width,labels=year)

plt.ylabel("AUC value")
plt.xlabel("k-flod")
plt.legend(loc="upper right")

plt.show()

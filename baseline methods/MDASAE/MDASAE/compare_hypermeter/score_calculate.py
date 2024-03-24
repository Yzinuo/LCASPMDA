import numpy as np

drug = np.loadtxt("./score3.txt")
n = 108



s = []
d = []
for i in range(1373):
    for j in range(173):
        if j == n:
            s.append(drug[i][j])
#np.sort(s)
print(np.sort(s))
d.extend(np.argsort(s))

print(d)
print("前20种微生物序号:")
for i in range(1353,len(d)):
    print(d[i])


#验证别的药物与相关微生物之间的联系
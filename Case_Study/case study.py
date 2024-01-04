import numpy as np
import pandas as pd

microbe=pd.read_excel("data/MDAD/microbes.xlsx")
microbe=microbe.values
disease=pd.read_excel("E:\实验室\L_CATSMDA\data\MDAD\drugs.xlsx")
disease=disease.values
score=np.loadtxt("data/MDAD/Score/SCORE/score0.txt")


disease1=list(score[598,:])     #Asthma
disease2=list(score[1105,:])     #IBS


md1=dict()
md2=dict()



for i in range(len(disease1)):
    md1[i]=disease1[i]
    md2[i]=disease2[i]




dm1=sorted(md1.items(),key=lambda x : x[1],reverse=True)
dm2=sorted(md2.items(),key=lambda x : x[1],reverse=True)

d1=[]
d2=[]
# d3=[]


for i in range(50):
    d1.append(microbe[dm1[i][0]][1])
    d2.append(microbe[dm2[i][0]][1])
    # d3.append(microbe[dm3[i][0]][1])

d1 = [s.replace('\xa0', ' ') for s in d1]
d2 = [s.replace('\xa0', ' ') for s in d2]
# d3 = [s.replace('\xa0', ' ') for s in d3]
np.savetxt("Case_Study/microbe1_drug.txt",d1,fmt="%s")
np.savetxt("Case_Study/microbe2_drug.txt",d2,fmt="%s")
# np.savetxt("Case_Study/microbe3_drug.txt",d3,fmt="%s")

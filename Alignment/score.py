import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn  import  preprocessing

X = pd.read_csv("E:/scGCN/atac.csv ",index_col = 0)   
X = X.values
X=X.T
Y = pd.read_csv('E:/scGCN/rna.csv',index_col = 0)  
Y = Y.values
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
Z = pd.concat([X,Y],axis=0)
Z = preprocessing.scale(Z)
Z=np.array(Z)
n_neighbors=23
nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(Z)
distances, indices = nbrs.kneighbors(Z)
indices = pd.DataFrame(indices)
m,n=indices.shape
l = int(m/2)
l1=int(X.shape[0])
l2=int(Y.shape[0])
a1=indices[:l1]
a2=indices[l1:]
for i in range(l1):
    for j in range(n):
        if  a1.iloc[i].iat[j]<l:
            a1.iloc[i].iat[j]=1
        else:
            a1.iloc[i].iat[j]=0
for i in range(l2):
    for j in range(n):
        if  a2.iloc[i].iat[j]>=l:
            a2.iloc[i].iat[j]=1
        else:
            a2.iloc[i].iat[j]=0
k = n_neighbors
b1=a1.sum()
b1=pd.DataFrame(b1)
b1=b1.sum()
b2=a2.sum()
b2=pd.DataFrame(b2)
b2=b2.sum()
x=b1+b2
x = (x/m)
h = x/k
aligement_score =1-((x-k/2)/(k/2))
print(aligement_score)




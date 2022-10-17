import nimfa
import pandas as pd


Z=pd.read_csv('E:/scGCN/rna.csv',index_col = 0)
Z=Z.values
Z=Z.T
nmf = nimfa.Nmf(Z, seed='random_vcol', rank=600, max_iter=100)
nmf_fit = nmf()
X_reduction=nmf_fit.fit.basis()
X_reduction=pd.DataFrame(X_reduction)
X_reduction=X_reduction.T
X_reduction.to_csv('E:/scGCN/trainnmf.csv')
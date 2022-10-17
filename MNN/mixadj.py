import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
import multiprocessing
import math



def NmaxIndex(test_list, N):

    return sorted(range(len(test_list)), key=lambda sub: test_list[sub])[-N:]
def queryN(rna_atac_metric, i, N):

    return NmaxIndex(list(rna_atac_metric.iloc[:, i].values), N)
def getQuery(rna_atac_metric, i, N):

    return queryN(rna_atac_metric, i, N)
def getQuery_parallel(rna_atac_metric, i, N):

    return [getQuery(rna_atac_metric, j, N) for j in i]
def query_helper(args):

    return getQuery_parallel(*args)
def parallel_query(rna_atac_metric, N1, N2, n_jobs=5):

    p = multiprocessing.Pool(n_jobs + 1)
    N_rows, N_cols = rna_atac_metric.shape
    njob1 = math.floor(n_jobs / 2)

    job_args1 = [(rna_atac_metric, i, N1) for i in np.array_split(range(N_cols), n_jobs - njob1)]
    job_args2 = [(rna_atac_metric.T, i, N2) for i in np.array_split(range(N_rows), njob1)]
    job_args = job_args1 + job_args2

    result = p.map(query_helper, job_args)
    result = flattenListOfLists2(result)
    p.close()

    return result
def flattenListOfLists2(lst):
    result = []
    [result.extend(sublist) for sublist in lst]  # uggly side effect ;)
    return result
def single_query(rna_atac_metric, N1):

    N_rows, N_cols = rna_atac_metric.shape #细胞基因
    result = [getQuery(rna_atac_metric, i, N1) for i in range(N_cols)]

    return result
def find_mutual_nn(rna_atac_metric, N1=3, N2=3, n_jobs=1):

    N_rows, N_cols = rna_atac_metric.shape
    index1 = rna_atac_metric.index
    index2 = rna_atac_metric.columns

    if n_jobs == 1:
        k_index_1 = single_query(rna_atac_metric, N1)
        k_index_2 = single_query(rna_atac_metric.T, N2)
    else:
        result = parallel_query(rna_atac_metric, N1, N2, n_jobs=n_jobs)
        k_index_1 = result[:N_cols]
        k_index_2 = result[N_cols:]

    mutual_1 = []
    mutual_2 = []
    for index_2 in range(N_cols):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)

    return mutual_1, mutual_2
def filterPairs(arr, similarity_rna_atac1, N1=2561, N2=2561, n_jobs=1):
    N_rows, N_cols = similarity_rna_atac1.shape

    if n_jobs < 2:
        k_index_1 = single_query(similarity_rna_atac1, N1)
        k_index_2 = single_query(similarity_rna_atac1.T, N2)
    else:
        result = parallel_query(similarity_rna_atac1, N1, N2, n_jobs=n_jobs)
        k_index_1 = result[:N_cols]
        k_index_2 = result[N_cols:]

    arr1 = np.array([0, 0])
    for i in range(arr.shape[0]):
        if (arr.iloc[i, 0] in k_index_1[arr.iloc[i, 1]]) and (arr.iloc[i, 1] in k_index_2[arr.iloc[i, 0]]):
            arr1 = np.vstack((arr1, arr.iloc[i, :]))
    arr1 = np.delete(arr1, (0), axis=0)

    return pd.DataFrame(arr1)
def selectPairs(df, similarity_matrix, N=3):

    weight_df = pd.DataFrame([[row[0], row[1],
                               similarity_matrix.iloc[row[0], row[1]]]
                              for index, row in df.iterrows()])

    g = []
    for i in range(2):
        g.append(weight_df.
                 groupby([i]).
                 apply(lambda x: x.sort_values([2], ascending=True)).
                 reset_index(drop=True).groupby([i]).head(N))

    g1 = pd.concat(g, ignore_index=True).drop_duplicates()

    return pd.concat(g, ignore_index=True).iloc[:, [0, 1]].drop_duplicates(), g1

X = pd.read_csv("E:/scGCN/atac.csv",index_col = 0)
X = X.values
a=X.shape[0]
similarity_selected = pd.DataFrame(
  pairwise_kernels(X,
                  X,
                  metric='cosine')
)
ref_pair, query_pair = find_mutual_nn(similarity_selected,
                                   N1=2,
                                   N2=2,
                                   n_jobs=1)

pair_ref_query = pd.DataFrame([ref_pair, query_pair]).T
pair_ref_query = filterPairs(pair_ref_query,
                                similarity_selected,
                                n_jobs=1)
pair_ref_query.drop_duplicates()
pair_ref_query, g1 = selectPairs(pair_ref_query,similarity_selected,
                                    N=2)
m=pair_ref_query.shape[0]
pair_ref_query = np.array(pair_ref_query)

indices = pair_ref_query
A = np.zeros(shape=(a,a))
for i in range(m):
    A[indices[i,0], indices[i,0]]=1
    A[indices[i,0], indices[i,1]]=A[indices[i,1], indices[i,0]]=1
A = pd.DataFrame(A)
a1=A.sum()
D1 = np.diag(a1**(-0.5))
D1 = pd.DataFrame(D1)
D1.to_csv("E:/scGCN/dujuzhen.csv")
A.to_csv("E:/scGCN/adj.csv")


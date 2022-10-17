# GCN-SC
A universal framework for single-cell multi-omics data integration with graph convolutional networks
A single-cell omics data integration algorithm based on graph convolutional neural networks. With the help of graph convolutional neural network, it can not only remove batch effects between different sequencing methods, omics, and species, but also explore the nonlinear relationship between cells in single-cell omics data and effectively integrate data.

###GCN-SC uses the following dependencies:
* python 3.7.10
*pytorch 1.8.0
* numpy 1.15.5
* scikit-learn 0.24.2
* pandas 1.3.1
* nimfa 1.4.0


###Guiding principles:

**MNN

  adj.py 
  
  Finding internal anchor pairs for query omics data
  
  mixadj.py 
  
  Finding anchor pairs between query omics and reference omics data

**scImpute

scimpute.R 

Imputed transcriptome data

**GCN

utils.py 

Definition of base class

layers.py 

Definition of the classes used by the model

models.py 

Definition of the model

train.py  

Running

**NMF:

nmf.py 

Dimensionality reduction algorithm




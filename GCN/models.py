import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np 
import torch

class GCN(nn.Module):
    def __init__(self,in_features,out_features,nclass,dropout):
        super(GCN, self).__init__()
        self.in_features=in_features
        self.out_features = out_features
        self.gcn1 = GraphConvolution(in_features, out_features)   
        self.gcn2 = GraphConvolution(out_features,out_features)
        self.dropout = dropout    
    def forward(self,adj,features):
        out = F.relu(self.gcn1(features,adj))   
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gcn2(out,adj)
        return out
    
    
class GCN_classifier(nn.Module):
    def __init__(self,in_features,out_features,nclass,dropout):
        super(GCN, self).__init__()
        self.in_features=in_features
        self.out_features = out_features
        self.gcn1 = GraphConvolution(in_features, out_features)   
        self.gcn2 = GraphConvolution(out_features,in_features)
        self.gcn3 = GraphConvolution(out_features,nclass)
        self.dropout = dropout    
    def forward(self,adj,features):
        out = F.relu(self.gcn1(features,adj))   
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gcn3(out,adj)
        out=F.softmax(out)
        return out

    


 
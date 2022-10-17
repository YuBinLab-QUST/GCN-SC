from __future__ import division
from __future__ import print_function

import time
import argparse  
import numpy as np
import torch
import torch.optim as optim
from utils import load_data
from models import GCN
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--in_features', type=int, default=2000,
                    help='number of features input')
parser.add_argument('--nclass', type=int, default=12,
                    help='class of cell')
parser.add_argument('--out_features', type=int, default=2000,
                    help='number of features output')


def train(epoch):
    t = time.time()  
    model.train()
    

    global output
    output= model(adj,features)
    output = torch.Tensor(output)
    mse_loss = torch.nn.MSELoss(size_average=True)
    loss_train = mse_loss(Y,output[label.long()]) 
    
    '''
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(Y,out[label.long()])
    '''
   
    optimizer.zero_grad()
    loss_train.requires_grad_(True)  
    loss_train.backward()  
    optimizer.step()  
    print(output)
    if not args.fastmode:
        model.eval() 
        output = model(adj, features)
        
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    
    
#load data
adj, features, z = load_data()
rna = pd.read_csv('E:/scGCN/rna.csv', index_col= 0)
rna = rna.values  
rna = rna.T  
z = torch.FloatTensor(rna)
atac = pd.read_csv('E:/scGCN/atac.csv',index_col= 0)
atac = atac.values
atac = atac.T
features = torch.FloatTensor(atac)
adj = pd.read_csv("E:/scGCN/adj.csv",index_col=0)
D=pd.read_csv("E:/scGCN/dujuzhen.csv",index_col=0)
adj = D*adj*D
adj=np.array(adj)
z = torch.FloatTensor(z)
adj = torch.FloatTensor(adj)
Y = pd.read_csv('E:/scGCN/threernaanchor.csv',index_col=0)
Y = Y.values
Y = torch.Tensor(Y)
Y1= pd.read_csv('E:/scGCN/threernaanchorone.csv',index_col=0)
Y1 = Y1.values
Y1 = torch.FloatTensor(Y1)
label = pd.read_csv('E:/scGCN/threelabel.csv',index_col =0)  

label = label.values
label=label.reshape(6175)
label=torch.Tensor(label)
label=label.T



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)   
if args.cuda:
    torch.cuda.manual_seed(args.seed) 

model = GCN(in_features=args.in_features,
            out_features = args.out_features,
            dropout=args.dropout)

'''
model = GCN_classifier(in_features=args.in_features,
            out_features = args.out_features,
            nclass = args.nclass
            dropout=args.dropout)
'''

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()   
    features = features.cuda()
    adj = adj.cuda()



t_total = time.time()
for epoch in range(args.epochs):
    t = time.time()  
    global output
    output= model(adj,features)
    output = torch.Tensor(output)    
    mse_loss = torch.nn.MSELoss(size_average=True)
    loss_train = mse_loss(Y,output[label.long()]) 
    '''
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(Y,out[label.long()])
    '''
    optimizer.zero_grad()
    loss_train.requires_grad_(True)  
    loss_train.backward()  
    optimizer.step() 
    if not args.fastmode:
        model.eval()  
        output = model(adj, features)
        
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

x = output.detach().numpy()
data=pd.DataFrame(x)
data.to_csv('E:/scGCN/out.csv')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

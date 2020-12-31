from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN, MLP
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=8,help='Propagation step')
parser.add_argument('--sample', type=int, default=3, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.0, help='Lamda')
parser.add_argument('--dataset', type=str, default='pubmed', help='Data set')
parser.add_argument('--cuda_device', type=int, default=7, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
A, adj, features, labels, idx_train, idx_val, idx_test, edges = load_data(dataset)
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)
# Model and optimizer
model = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn = args.use_bn)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()
    A = A.cuda()

def propagate(feature, a, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    #x = feature
    y = torch.spmm(a,feature).detach_()
    x = y
    for i in range(order):
        if i ==0:
            x = torch.spmm(a, x).detach_()
        else:
            x = torch.spmm(a, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        #y= x
    return y.div_(order+1.0).detach_()
def sparse_dropout(a, training, dropedge_rate):
    indice = a._indices()
    values = a._values() 
    values = F.dropout(values, p=dropedge_rate, training=training)
    size = a.size()
    a = torch.sparse.FloatTensor(indice, values, size)
    #d1 = torch.diag(a.sum(dim=1)**(-0.5))
    #d2 = torch.diag(a.sum(dim=0)**(-0.5))
    
    return a
def preprocess(a):
    #d1 = np.array(a.sum(axis-1))**(-0.5)
    #d2 = np.array(a.sum(axis=0))**(-0.5)
    D1_ = np.array(a.sum(axis=1))**(-0.5)
    D2_ = np.array(a.sum(axis=0))**(-0.5)
    D1_ = sp.diags(D1_[:,0], format='csr')
    D2_ = sp.diags(D2_[0,:], format='csr')
    A_ = a.dot(D1_)
    A_ = D2_.dot(A_)
    A_ = sparse_mx_to_torch_sparse_tensor(A_) 
    if args.cuda:
        A_ = A_.cuda()
    return A_    


def random_edge_sample(edges, droprate):
    edges = list(edges)
    n = features.shape[0]
    m = len(edges)
    index = np.random.permutation(m)
    percent = 1. - droprate
    preserve_num = int(m * percent)
     
    index_ = index[:preserve_num]
    sample_row = [edges[x][0] for x in index_]
    sample_col = [edges[x][1] for x in index_]
    sample_adj = sp.csr_matrix((np.ones(preserve_num), (sample_row, sample_col)), shape=(n,n))
    sample_adj = sample_adj + sample_adj.T.multiply(sample_adj.T>sample_adj) - sample_adj.multiply(sample_adj.T>sample_adj) + sp.eye(n)
    sample_adj = preprocess(sample_adj)
    """
    nnz = a.nnz
    percent = 1. - droprate
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[: preserve_nnz]
    r_adj = sp.coo_matrix((a.data[perm],
                           (a.row[perm],
                            a.col[perm])),
                            shape=a.shape)
     
    print(r_adj.nnz, #a.nnz)
    r_adj = preprocess(r_adj)
    """
    return sample_adj                          
    
    
def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    #drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    if training: 
        a = random_edge_sample(edges, drop_rate)
        #a = sparse_dropout(A, training, drop_rate)
    else:
        a = A#preprocess(adj)
    features = propagate(features, a, args.order)    
    return features

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def train(epoch):
    t = time.time()
    
    X = features
    
    model.train()
    optimizer.zero_grad()
    X_list = []
    K = args.sample
    for k in range(K):
        X_list.append(rand_prop(X, training=True))

    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

    
    loss_train = 0.
    for k in range(K):
        loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
     
        
    loss_train = loss_train/K
    #loss_train = F.nll_loss(output_1[idx_train], labels[idx_train]) + F.nll_loss(output_1[idx_train], labels[idx_train])
    #loss_js = js_loss(output_1[idx_unlabel], output_2[idx_unlabel])
    #loss_en = entropy_loss(output_1[idx_unlabel]) + entropy_loss(output_2[idx_unlabel])
    loss_consis = consis_loss(output_list)

    loss_train = loss_train + loss_consis
    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        X = rand_prop(X,training=False)
        output = model(X)
        output = torch.log_softmax(output, dim=-1)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()
def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset +'.pkl'))



def test():
    model.eval()
    X = features
    X = rand_prop(X, training=False)
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
Train()
test()

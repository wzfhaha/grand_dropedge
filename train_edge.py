from __future__ import division
from __future__ import print_function

import glob
import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, MMD, label_vector_generator, load_data_p, sparse_mx_to_torch_sparse_tensor
from pygcn.models import GCN, NGCN
import scipy.sparse as sp

#from pygat.models import SpGAT

# from gatlabel.models import LabelGat, SpGAT

from center_loss_soft import CenterLoss

from sklearn.metrics import f1_score


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,  # 10000
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hid1', type=int, default=32,
                    help='Number of hidden units 1.')
parser.add_argument('--hid2', type=int, default=7,
                    help='Number of hidden units 2.')
parser.add_argument('--nheads', type=int, default=1,
                    help='nheads 8.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=2, help='Patience')
parser.add_argument('--alpha', type=float, default=0.1, help='Patience')

args = parser.parse_args()

def Feature_Generator_(adj, features, order, alpha=args.alpha):
    s = features
    r = features
    for i in range(order):
        r = torch.spmm(adj, r*(-alpha)).detach_()
        s.add_(r)
    del r
    return s.detach_()

def Edge_Generator_(adj, order, alpha=args.alpha):
    s = adj
    r = adj
    for i in range(order):
        r = r * (-alpha)
        s.add_(r)
    del r
    return s


def Edge_Generator(adj, order, alpha=args.alpha, ):
    s = adj
    r = adj.to_dense()
    s2 = adj
    for i in range(order):
        #print(r)
        r = torch.spmm(adj, (r* alpha))
        #truc_r = r.copy()
        #truc_r.data = np.ones(len(truc_r.data))
        #truc_r = truc_r - s
        s = (r +s).to_sparse()
        #s2 += alpha * truc_r
    del r
    #s.data[s.data>0.5] = 1.
    #s.data[s.data<=0.5] = 0.
    #print(sum(s.data), len(s.data))
    #print(np.sum(s.todense()))
    #s[np.where[s>0.5]] = 1.
    #s[np.where[s<=0.5]] = 0.
    return s

def Feature_Generator(adj, features, order=0):
    # features_0 = [features]
    s = features
    r = features


    # D = torch.spmm(adj, torch.ones[n, 1])**(-0.5)
    # A = D*A
    # A = torch.squeeze(D)*A

    for i in range(order):
        r = torch.spmm(adj, r).detach_()
        s.add_(r)
    del r
        # features_0.append(alpha*torch.spmm(adj, features_0[-1]) + (1-alpha)*features_0[0])
    return s.div_(order+1.0).detach_()

def adj2A(adj):
    adj = adj + sp.eye(adj.shape[0])
    D1 = np.array(adj.sum(axis=1))**(-0.5)
    D2 = np.array(adj.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:,0], format='csr')
    D2 = sp.diags(D2[0,:], format='csr')
    A = adj.dot(D1)
    A = D2.dot(A)
    #A = sp.eye(adj.shape[0]) - A
    A = sparse_mx_to_torch_sparse_tensor(A)
    return  A

def feature_generator_(adj, A, features, order, edges, alpha=args.alpha):
    n = features.shape[0]
    edges = list(edges)
    m = len(edges)
    index = np.random.permutation(m)
    #print(index[:10])
    index_1 = index[: m//2]
    index_2 = index[m//2 :]
    mask_1_row = [edges[x][0] for x in index_1]
    mask_1_col = [edges[x][1] for x in index_1]
    mask_2_row = [edges[x][0] for x in index_2]
    mask_2_col = [edges[x][1] for x in index_2]
    
    mask_1 = sp.csr_matrix((np.ones(m//2), (mask_1_row, mask_1_col)), shape=(n, n))
    mask_2 = sp.csr_matrix((np.ones(m//2), (mask_2_row, mask_2_col)), shape=(n, n))
    adj_1 = mask_1 + mask_1.T.multiply(mask_1.T>mask_1) - mask_1.multiply(mask_1.T>mask_1) + sp.eye(n)
    adj_2 = mask_2 + mask_2.T.multiply(mask_2.T>mask_2) - mask_2.multiply(mask_2.T>mask_2) + sp.eye(n)
    
    adj_1 = sparse_mx_to_torch_sparse_tensor(adj_1)
    adj_2 = sparse_mx_to_torch_sparse_tensor(adj_2)
    if args.cuda:
        adj_1 = adj_1.cuda()
        adj_2 = adj_2.cuda()
    A_1 = A * adj_1
    A_2 = A * adj_2

    A_1 = Edge_Generator(A_1, args.order)
    A_2 = Edge_Generator(A_2, args.order)
    #A_1 = Edge_Generator(A_1, args.order)
    #A_2 = Edge_Generator(A_2, args.order)
    #if args.cuda:
    #    A_1 = A_1.cuda()
    #    A_2 = A_2.cuda()
    #mask_1[index_1] = 1
    #mask_2[index_2] = 1
    
    #features_1 = mask_1.cuda() * features
    #features_2 = mask_2.cuda() * features
     
    # D = torch.spmm(adj, torch.ones[n, 1])**(-0.5)
    # A = D*A
    # A = torch.squeeze(D)*A
    #alpha = 1
    #features_1 = Feature_Generator(A_1, features, order)
    #features_2 = Feature_Generator(A_2, features, order)
    return A_1, A_2


c_ctr = 1.

def theloss(outputs, idx):
    r_1, r_2, y_1, y_2 = outputs

    p_1 = torch.exp(y_1)
    p_2 = torch.exp(y_2)

    # torch.cat((emb[idx_val], emb[idx_test]), 0))
    # l_ctr = center_loss_1(r_1, p_2.detach()) + center_loss_2(r_2, p_1.detach())
    l_kl = (torch.mean(p_1 * (y_1 - y_2)) + torch.mean(p_2 * (y_2 - y_1))) *0.5
    # l_dis = MMD(y_1, y_2)
    l_class = (F.nll_loss(y_1[idx], labels[idx]) + F.nll_loss(y_2[idx], labels[idx]))*0.5

    return c_ctr*l_kl + l_class


def theloss_cold(outputs, idx):
    r_1, r_2, y_1, y_2 = outputs

    p_1 = torch.exp(y_1)
    p_2 = torch.exp(y_2)

    # torch.cat((emb[idx_val], emb[idx_test]), 0))
    # l_ctr = center_loss_1(r_1[idx].detach(), p_1[idx].detach()) + center_loss_2(r_2[idx].detach(), p_2[idx].detach())
    l_kl = (torch.mean(p_1 * (y_1 - y_2)) + torch.mean(p_2 * (y_2 - y_1))) *0.5
    # l_dis = MMD(y_1, y_2)
    l_class = (F.nll_loss(y_1[idx], labels[idx]) + F.nll_loss(y_2[idx], labels[idx]))*0.5

    return c_ctr*l_kl + l_class


# import sys
 
# class Logger(object):
#     def __init__(self, fileN="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(fileN, "w")
 
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
 
#     def flush(self):
#         pass
 
# sys.stdout = Logger("target_file.txt")



args.cuda = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.cuda.set_device(5)

print('args.cuda', args.cuda)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args.seed)
# Load data
dataset = 'cora'
#dataset = 'citeseer'
# dataset = 'pubmed'
adj, A, _,features, labels, idx_train, idx_val, idx_test, edges, adj_ = load_data(dataset)

model = NGCN(nfeat_1=features.shape[1], 
             nfeat_2=features.shape[1],
             nhid_1=args.hid1,
             nhid_2=args.hid1,
             nclass=labels.max().item() + 1, 
             dropout=args.dropout)
            # dropout=0.5)

# center_loss_1 = CenterLoss(num_classes=labels.max().item() + 1, feat_dim=16, use_gpu=args.cuda)
# center_loss_2 = CenterLoss(num_classes=labels.max().item() + 1, feat_dim=16, use_gpu=args.cuda)

params = list(model.parameters()) #+ list(center_loss_1.parameters()) + list(center_loss_2.parameters())

optimizer = optim.Adam(params,
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    features = features.cuda()
    # label_vec = label_vec.cuda()
    adj = adj.cuda()
    A = A.cuda()
    #A2 = A2.dot_(A2)
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    model.cuda()

def train(epoch, cold_start=False):
    model.train()

    t = time.time()
    #A_1, A_2 = A,A
    A_1, A_2 = feature_generator_(adj_ , A, features, args.order, edges)
    x = features
    optimizer.zero_grad()
    outputs = model(x, x, A_1, A_2)

    A2 = Edge_Generator(A * 0.5, args.order, args.alpha)
    #print(np.sum((adj_).todense()))
    #A = adj2A(A)
    #if args.cuda:
    #    A = A.cuda()
    if cold_start:
      loss_train = theloss_cold(outputs, idx_train) #labels_2
    else:
      loss_train = theloss(outputs, idx_train)
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        outputs = model(x, x, A2, A2)

    if cold_start:
      loss_val = theloss_cold(outputs, idx_val)
    else:
      loss_val = theloss(outputs, idx_val)


    # prob = torch.cat([outputs[-1], outputs[-2]], 1)
    prob = (torch.exp(outputs[-1]) + torch.exp(outputs[-2]))/2.0

    # y_true = labels.cpu().data.numpy()
    # y_pred = (prob.max(1)[1]%(labels.max().item() + 1)).type_as(labels).cpu().data.numpy()

    # acc_train = f1_score(y_true[idx_train], y_pred[idx_train], average='micro')
    # acc_val = f1_score(y_true[idx_val], y_pred[idx_val], average='micro')
    acc_train = accuracy(prob[idx_train], labels[idx_train])
    acc_val = accuracy(prob[idx_val], labels[idx_val])

    if (epoch+1)%1 == 0:
      # print('num of training', len(idx_train_fake))
      print('{:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.5f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.5}'.format(acc_val.item()),
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
    #x = Feature_Generator(A, features, args.order)
    #x = Feature_Generator(A, features, args.order)
    x = features
    A2 = Edge_Generator(A*0.5, args.order, args.alpha)
    #A = adj2A(A)
    #if args.cuda:
    #    A = A.cuda()
    outputs = model(x, x, A2, A2)

    loss_test = theloss(outputs, idx_test)

    prob = (torch.exp(outputs[-1]) + torch.exp(outputs[-2]))/2.0
    # prob = torch.cat([outputs[-1], outputs[-2]], 1)

    # y_true = labels.cpu().data.numpy()
    # y_pred = (prob.max(1)[1]%(labels.max().item() + 1)).type_as(labels).cpu().data.numpy()
    # print(y_pred[:10])
    # assert False
    # acc_test = f1_score(y_true[idx_test], y_pred[idx_test], average='micro')
    # acc_val = f1_score(y_true[idx_val], y_pred[idx_val], average='micro')

    acc_test = accuracy(prob[idx_test], labels[idx_test])
    y_true = labels[idx_test].cpu().data.numpy()
    y_pred = prob[idx_test].max(1)[1].type_as(labels[idx_test]).cpu().data.numpy()
    f1_test = f1_score(y_true, y_pred, average='micro')
    print("Model Test set results:",
          "test_loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.5f}".format(acc_test.item()),
          "f1_score= {:.5f}".format(f1_test))

Train()
test()



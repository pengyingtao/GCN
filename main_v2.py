# encoding:utf-8
import sys, os
import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

from dgl.data import citation_graph as citegrh

import time
import warnings
import numpy as np


# gcn的聚合函数和reduce
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}
    
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)  # 结点的聚合与消息传递
        g.apply_nodes(func=self.apply_mod)  # 线性变换
        return g.ndata.pop('h')
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x

# 从库中load data
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask


if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # 忽略warning

    # model的结构
    GCnet = Net()
    print(GCnet)

    # 图，node's 特征，标签， 加载数据
    graph, features, labels, mask = load_cora_data()

    # 优化操作
    optimizer = th.optim.Adam(GCnet.parameters(), lr=0.1)

    dur = []
    train_loss = []

    for epoch in range(100):
        if epoch >= 3:
            t0 = time.time()
        
        logits = GCnet(graph,  features)
        logp = F.log_softmax(logits, 1)  # softmax 变成最后的分类问题
        loss = F.nll_loss(logp[mask], labels[mask])  # 计算反馈loss

        # 初始化梯度
        optimizer.zero_grad()
        # loss反馈
        loss.backward()
        # 优化step
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
            train_loss.append(loss.item())
        
        print("Epoch %5d  |  Loss %.4f  |  Time(s) %.4f"%(epoch, loss.item(), np.mean(dur)))

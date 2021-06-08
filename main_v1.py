# encoding:utf-8
import dgl
import os, sys
import networkx as nx
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation


# 构建graph数据
def build_karate_club_graph():
    g = dgl.DGLGraph()
    # 添加34个结点， node的label为0-33
    g.add_nodes(34)
    # 添加78个edge，用tuples的list类型
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]

    # 添加两个节点列表：src和dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # 构造双向边
    g.add_edges(dst, src)
    return g


def graph_plot(G):
    fig = plt.figure(dpi=150)
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


# 主要定义message和reduce方法
# Note： 为了
def gcn_message(edges):
    # 参数：batch of edges
    # 得到计算后的batch of edges的信息，这里直接返回边的源节点的feature.
    return {'msg': edges.src['h']}


def gcn_reduce(nodes):
    # 参数：batch of nodes.
    # 得到计算后batch of nodes的信息，这里返回每个节点mailbox里的msg的和
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}


    
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
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


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


# define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        # self.linear = nn.Linear(in_feats, out_feats)
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, inputs):
        # g为图对象，inputs为节点的特征矩阵
        # 设置图的结点特征
        # print(inputs)
        g.ndata['h'] = inputs
        
        # 触发边的信息传递
        # g.send(g.edges(), gcn_message)
        # 触发节点的聚合函数
        # g.recv(g.nodes(), gcn_reduce)

        g.update_all(gcn_message, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)   # 中间加了线性变换Linear

        # 取得节点向量
        h = g.ndata.pop('h')

        return h


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, F.relu)
        self.gcn2 = GCNLayer(hidden_size, num_classes, F.relu)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)  # 第一层将34层的输入转化为隐层为8
        h = torch.relu(h)
        h = self.gcn2(g, h)      # 第二层将隐层转化为最终的分类数2
        return h


if __name__ == '__main__':
    G = build_karate_club_graph()
    G.ndata['feat'] = torch.eye(34)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    # 画图
    # graph_plot(G)

    # 以空手道俱乐部为例
    net = GCN(34, 8, 2)
    print(net)  # 打印网络结构

    inputs = torch.eye(34)   # 34个结点 ont-hot表示  # tensor([[1., 0., 0.,  ..., 0., 0., 0.],....
    # 仅有指导员（节点0）和俱乐部主席（节点33）被分配了label
    labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled  # tensor([ 0, 33])
    labels = torch.tensor([0, 1])  # tensor([0, 1])

    '''
    创建优化器
    输入inputs
    计算loss
    使用反向传播
    '''
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 创建优化器
    
    all_logits = []  # 日志
    # nx_G = G.to_networkx().to_undirected()
    # # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    # pos = nx.kamada_kawai_layout(nx_G)

    for epoch in range(100):  # 设置50个epoch
        logits = net(G, inputs)

        # we save the logits for visualization later
        all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


    # fig = plt.figure(dpi=150)
    # fig.clf()
    # ax = fig.subplots()
    # # draw(0)  # draw the prediction of the first epoch
    # ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    # plt.pause(30)
    # pass

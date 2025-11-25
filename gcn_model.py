import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
# from layer import GraphConvolution
from layer import ModifiedGCN
"""
class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)

        x = self.fc(x)

        return x
"""

# DeepGCN
class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_blocks, dropout):
        super(GCN, self).__init__()
        # 使用ModifiedGCN替换原有的层
        self.modified_gcn = ModifiedGCN(n_in, n_hid, n_out, n_blocks, dropout)

    def forward(self, data):
        # 直接调用ModifiedGCN的forward方法
        # features = data['x']
        # adj = data['edge_index']
        # return self.modified_gcn({'x': features, 'edge_index': adj})

        return self.modified_gcn(data)

class multiGATModelAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, npatient, n_blocks, dropout=0.6, alpha=0.1,):
        super(multiGATModelAE, self).__init__()
        self.dropout = dropout
        # multiGATmodelAE中 Multi-Head-Attention 是根据实例数目来设置的，model传入两个实例
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        # create attention layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        # ModifiedGCN part with nhid as input feature dimension
        self.modified_gcn = ModifiedGCN(nhid, nhid, nclass, n_blocks, dropout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)

        # Pass the output from GAT to ModifiedGCN
        data = {'x': x, 'edge_index': adj}
        x = self.modified_gcn(data)
        return x




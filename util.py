import torch
from sklearn import metrics
import pandas as pd
import numpy as np

def load_data(adj, fea, phy, lab, threshold=0.005):
# def load_data(adj, fea, lab, threshold=0.005):
# def load_data(adj, lab, threshold=0.005):
    '''
    :param adj: the similarity matrix filename
    :param fea: the omics vector features filename
    :paarm phy: the phylogeneticTree feature filename
    :param lab: sample labels  filename
    :param threshold: the edge filter threshold
    '''
    print('loading data...')
    #无系统发育树/自编码器
    # adj_df = pd.read_csv(adj, header=0, index_col=None)
    # fea_df = pd.read_csv(adj, header=0, index_col=None)
    # label_df = pd.read_csv(lab, header=0, index_col=None)

    #系统发育树/自编码器
    # adj_df = pd.read_csv(adj, header=0, index_col=None)
    # fea_df = pd.read_csv(fea, header=0, index_col=None)
    # # fea_df = pd.concat([fea_df.iloc[:, :], adj_df.iloc[:, 1:]], axis=1)
    # fea_df = pd.concat([adj_df.iloc[:, :], fea_df.iloc[:, 1:]], axis=1)
    # label_df = pd.read_csv(lab, header=0, index_col=None)

    #系统发育树+自编码器
    adj_df = pd.read_csv(adj, header=0, index_col=None)
    laten_df = pd.read_csv(fea, header=0, index_col=None)
    phy_df = pd.read_csv(phy, header=0, index_col=None)
    label_df = pd.read_csv(lab, header=0, index_col=None)

#===============================================================
    # print("before:", adj_df.shape)
    # # Load the outlier samples to be removed
    # outlier_samples_df = pd.read_csv("./error_samples1.csv", header=0)
    # outlier_samples = outlier_samples_df['Sample'].values
    #
    # adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    # laten_df.rename(columns={laten_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    # phy_df.rename(columns={phy_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    # label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    #
    # adj_df = adj_df[~adj_df['Sample'].isin(outlier_samples)]
    # adj_df = adj_df.drop(columns=outlier_samples, errors='ignore')
    # laten_df = laten_df[~laten_df['Sample'].isin(outlier_samples)]
    # phy_df = phy_df[~phy_df['Sample'].isin(outlier_samples)]
    # label_df = label_df[~label_df['Sample'].isin(outlier_samples)]
    # print("after:", adj_df.shape)
    # print("laten_df_after:", laten_df.shape)
    # print("phy_df_after:", phy_df.shape)
    #
    # # fea_df = pd.concat([laten_df.iloc[:, :], adj_df.iloc[:, 1:], phy_df.iloc[:, 1:]], axis=1)
    # fea_df = pd.merge(phy_df, laten_df, on='Sample', how='inner')
    # fea_df = pd.merge(fea_df, adj_df, on='Sample', how='inner')
    # # fea_df = pd.concat([adj_df.iloc[:, :], phy_df.iloc[:, 1:],laten_df.iloc[:, 1:]], axis=1)
    # # print(fea_df)
#===============================================================

    fea_df = pd.concat([phy_df.iloc[:, :], laten_df.iloc[:, 1:],adj_df.iloc[:, 1:]], axis=1)

    if adj_df.shape[0] != fea_df.shape[0] or adj_df.shape[0] != label_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    # 将列名转化为列表，然后通过索引[0]访问列表中的第一个元素，即第一列列名
    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)

    #align samples of different data
    # adj_df.sort_values(by='Sample', ascending=True, inplace=True)
    # fea_df.sort_values(by='Sample', ascending=True, inplace=True)
    # label_df.sort_values(by='Sample', ascending=True, inplace=True)

    # print(fea_df)


    print('Calculating the laplace adjacency matrix...')
    adj_m = adj_df.iloc[:, 1:].values
    #The SNF matrix is a completed connected graph, it is better to filter edges with a threshold
    adj_m[adj_m<threshold] = 0

    # adjacency matrix after filtering
    exist = (adj_m != 0) * 1.0
    np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')

    #calculate the degree matrix
    factor = np.ones(adj_m.shape[1])
    res = np.dot(exist, factor)     #degree of each node
    diag_matrix = np.diag(res)  #degree matrix
    #np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')

    #calculate the laplace matrix
    d_inv = np.linalg.inv(diag_matrix)
    adj_hat = d_inv.dot(exist)

    return adj_hat, fea_df, label_df


def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
# def accuracy(preds, labels):
#     correct = preds.eq(labels).double().sum()
#     return correct / len(labels)

# if __name__ == "__main__":
#     adj, data_feat, label = load_data('./result/matrix_IBD.csv', './result/latent_IBD.csv','./result/PhylogenTree_p_IBD.csv', './data/labels_IBD.csv', threshold=0.005)




import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, auc, recall_score, precision_recall_curve, roc_curve,accuracy_score
import torch
import torch.nn.functional as F
from gcn_model import GCN
from gcn_model import multiGATModelAE
from util import load_data
from util import accuracy
from feature_selection_test import feature_select
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import re
import time
start_time = time.time()

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(epoch, optimizer, features, adj, labels, idx_train):
    '''
    :param features: the  features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_train: the index of trained samples
    '''
    labels.to(device)

    GCN_model.train()                      #将模型设置为训练模式
    optimizer.zero_grad()                  #将优化器的梯度清零

    # ##### 使用DeepGCN(残差网络) 建模
    # 将邻接矩阵转换为 edge_index
    edge_index, _ = dense_to_sparse(adj)   #将稠密的邻接矩阵 adj 转换为稀疏的边索引
    # 确保 edge_index 是整数类型
    edge_index = edge_index.to(torch.long)
    data = {'x': features, 'edge_index': edge_index}  #将特征和边索引组合成一个字典传递给 GCN 模型
    output = GCN_model(data)   #将数据传递给 GCN 模型，进行前向传播，输出为每个节点的预测结果
    ## 使用3层GCN特征学习和分类
    # output = GCN_model(features, adj)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])  #计算交叉熵损失，只计算训练集中的损失

    acc_train = accuracy(output[idx_train], labels[idx_train])          #计算训练样本的准确率
    softmax_output = F.softmax(output[idx_train], dim=1)                #对模型的输出进行 softmax 操作，这里也就是预测（取两类中概率值大的）
    try:                                                                #计算 AUC
        auc_train = roc_auc_score(labels[idx_train].cpu(), softmax_output[:, 1].detach().cpu())
    except ValueError:
        # 无法计算 AUC（可能因为类别不平衡或只有一个类别的样本）
        auc_train = float('nan')
    loss_train.backward()                                               #执行反向传播，计算损失函数相对于模型参数的梯度
    optimizer.step()                                                    #根据计算的梯度更新模型的参数
    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch: {epoch + 1:.2f} | Loss Train: {loss_train.item():.4f} | Acc Train:  {acc_train:.4f} | AUC Train: {auc_train:.4f}')
    return loss_train.data.item()


def test(features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    '''
    GCN_model.eval()
    ##### 使用GCN模型的架构
    # output = GCN_model(features, adj)

    ##### 使用残差Deep
    # 将邻接矩阵转换为 edge_index
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(torch.long)
    # 构建包含 features 和 edge_index 的字典
    data = {'x': features, 'edge_index': edge_index}
    output = GCN_model(data)  # 修改此处以使用 data 字典


    ##### GAT
    # output, _ = GCN_model(features, adj)  # 假设 features 是输入特征

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    # Loss = nn.BCEWithLogitsLoss()
    # loss_test = Loss(output[idx_test], labels[idx_test].view(-1, 1).float())
    #calculate the accuracy
    acc_test = accuracy(output[idx_test], labels[idx_test])

    #output is the one-hot label
    ot = output[idx_test].detach().cpu().numpy()
    #change one-hot label to digit label，预测得到标签
    ot = np.argmax(ot, axis=1)
    #original label
    lb = labels[idx_test].detach().cpu().numpy()
    print('predict label: ', ot)
    print('original label: ', lb)

    #calculate the f1 score
    f1 = f1_score(lb, ot, average='weighted')
    ##################################
    # 计算预测标签，其实就是ot
    predicted_labels = output[idx_test].detach().cpu().numpy()
    predicted_labels = np.argmax(predicted_labels, axis=1)

    # 计算TPR和FPR
    fpr, tpr, thresholds = roc_curve(labels[idx_test].cpu(), predicted_labels)

    # 将TPR和FPR写入文件
    with open("TPR_FPR.txt", 'w') as f:
        f.write("TPR: " + str(tpr.tolist()) + "\n")
        f.write("FPR: " + str(fpr.tolist()) + "\n")

    # Calculate AUC
    # Ensure that 'ot' contains probability scores
    auc_score = roc_auc_score(lb, ot, multi_class='ovr')  # Adjust parameters as needed

    # Calculate AUPR
    precision, recall, _ = precision_recall_curve(lb, ot)
    aupr = auc(recall, precision)

    # Calculate Recall and Precision
    rec = recall_score(lb, ot, average='weighted')
    prec = precision_score(lb, ot, average='weighted')

    # Print all metrics
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1_score= {:.4f}".format(f1),
          "AUC= {:.4f}".format(auc_score),
          "AUPR= {:.4f}".format(aupr),
          "Recall= {:.4f}".format(rec),
          "Precision= {:.4f}".format(prec))

    # Return all metrics
    return acc_test.item(), f1, auc_score, aupr, rec, prec



def predict(features, adj, sample, idx, label):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param sample: all sample names
    :param idx: the index of predict samples
    :return:
    '''
    GCN_model.eval()

    # ##### 使用DeepGCN 架构
    # 将邻接矩阵转换为 edge_index
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(torch.long)
    # 构建包含 features 和 edge_index 的字典
    data = {'x': features, 'edge_index': edge_index}
    output = GCN_model(data)  # 修改此处以使用 data 字典

    ##### 使用GCN架构
    # output = GCN_model(features, adj)
    ##### 使用GAT
    # output, _ = GCN_model(features, adj)

    # predict_label = output[idx].detach().cpu().numpy()
    predict_proba = torch.softmax(output[idx], dim=1).detach().cpu().numpy()
    predict_label = np.argmax(predict_proba, axis=1).tolist()
    print(predict_label)
    true_label = label[idx].detach().cpu().numpy()
    print(true_label)

    auc_score = roc_auc_score(true_label, predict_proba[:,0])  # 类别 1 的概率
    # 计算 ACC
    acc_score = accuracy_score(true_label, predict_label)
    # 计算 F1 分数
    f1 = f1_score(true_label, predict_label)
    # 计算 Recall
    recall = recall_score(true_label, predict_label)
    # 计算 Precision
    precision = precision_score(true_label, predict_label)
    # 打印结果
    print(f'AUC Score: {auc_score:.4f}')
    print(f'Accuracy Score: {acc_score:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')

    # res_data = pd.DataFrame({'Sample':sample, 'predict_label':predict_label, 'predict_proba':predict_proba1})
    # res_data = res_data.iloc[idx,:]
    # print(res_data)
    # res_data.to_csv('result/GCN_predicted_WT2D.csv', header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, default="./result/latent_mergedCWT2D.csv", help='The vector feature file.')
    parser.add_argument('--phylogeneTreedata', '-ph', type=str, default="./newdata/phylogenTree_p_mergedCWT2D.csv", help='The vector phylogenetic Tree file.')
    parser.add_argument('--adjdata', '-ad', type=str, default="./Similarity/fused_mergedCWT2D_matrix.csv", help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, default="./newdata/labels_mergedCWT2D.csv", help='The sample label file.')
    parser.add_argument('--testsample', '-ts', type=str, help='Test sample names file.',default=None)
    parser.add_argument('--mode', '-m', type=int, choices=[0,1], default=0,
                        help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--seed', '-s', type=int, default=1421, help='Random seed, default=123')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: gpu.')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='Training epochs, default: 500.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--hidden', '-hd',type=int, default=64, help='Hidden layer dimension, default: 64.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.4, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--threshold', '-t', type=float, default=0.004, help='Threshold to filter edges, default: 0.005') # 注意
    parser.add_argument('--nclass', '-nc', type=int, default=2, help='Number of classes, default: 2')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    args = parser.parse_args()

    # Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)


    # load input files
    # adj, data, label = load_data(args.adjdata, args.labeldata, args.threshold)
    # adj, data, label = load_data(args.adjdata, args.featuredata, args.labeldata, args.threshold)
    adj, data, label = load_data(args.adjdata, args.featuredata, args.phylogeneTreedata, args.labeldata, args.threshold)
    # change dataframe to Tensor
    print(label)
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')

    # 10-fold cross validation
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        # acc_res, f1_res = [], []  #record accuracy and f1 score
        # Initialize lists to record all metrics
        acc_res, f1_res, auc_res, aupr_res, recall_res, precision_res = [], [], [], [], [], []
        with open('Cross_Validation_results.txt', 'w') as f:
            fold = 0
            f.write("Fold\tAccuracy\tF1 Score\tAUC\tAUPR\tRecall\tPrecision\n")  # Write headers
            # split train and test data
            for idx_train, idx_test in skf.split(data.iloc[:, 1:], label.iloc[:, 1]):
                # ==================================================
                # train_features = features[idx_train]
                # train_labels = labels[idx_train]
                # # 使用 feature_select 函数在训练集上进行特征选择
                # selected_features_index = feature_select(train_features.cpu().numpy(), train_labels.cpu().numpy(),650)
                # # 应用选择的特征到训练集和测试集
                # train_features_selected = train_features[:, selected_features_index]
                # test_features_selected = features[idx_test][:, selected_features_index]
                # # 创建新的 features 矩阵，将选出的特征整合回去
                # features= torch.zeros((features.shape[0], len(selected_features_index)), dtype=torch.float, device=device)
                # features[idx_train] = train_features_selected
                # features[idx_test] = test_features_selected
                # ==================================================

                # 使用GCN搭建网络
                # GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)

                # 使用DeepGCN搭建网络
                GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, n_blocks=4, dropout=args.dropout)

                # 使用GAT搭建网络
                # # 初始化 multiGATModelAE 模型
                # nfeat = features.shape[1]  # 假设 features 是输入特征
                # nhid = args.hidden
                # nclass = args.nclass
                # dropout = args.dropout
                # alpha = 0.2  # 例如，您可以自定义这个值
                # nheads = 4  # 头的数量，根据需要调整
                # npatient = data.shape[0]  # 病人数，根据数据集调整
                #
                # GCN_model = multiGATModelAE(nfeat, nhid, nclass, dropout, alpha, nheads, npatient)

                GCN_model.to(device)

                # define the optimizer
                optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

                idx_train, idx_test= torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test, dtype=torch.long, device=device)
                #在训练集上训练指定批次的模型
                for epoch in range(args.epochs):
                    train(epoch, optimizer, features, adj, labels, idx_train)

                # calculate the accuracy and f1 score
                # ac, f1= test(features, adj, labels, idx_test)
                # acc_res.append(ac)
                # f1_res.append(f1)
                ac, f1, auc_score, aupr, recall, precision = test(features, adj, labels, idx_test)
                acc_res.append(ac)
                f1_res.append(f1)
                auc_res.append(auc_score)
                aupr_res.append(aupr)
                recall_res.append(recall)
                precision_res.append(precision)
                fold=fold+1
                f.write(f"{fold}\t{ac:.4f}\t{f1:.4f}\t{auc_score:.4f}\t{aupr:.4f}\t{recall:.4f}\t{precision:.4f}\n")
            # Print average and standard deviation for each metric
        print('10-fold Cross Validation Results:')
        print('Accuracy: Mean=%.4f, Std=%.4f' % (np.mean(acc_res), np.std(acc_res)))
        print('F1 Score: Mean=%.4f, Std=%.4f' % (np.mean(f1_res), np.std(f1_res)))
        print('AUC: Mean=%.4f, Std=%.4f' % (np.mean(auc_res), np.std(auc_res)))
        print('AUPR: Mean=%.4f, Std=%.4f' % (np.mean(aupr_res), np.std(aupr_res)))
        print('Recall: Mean=%.4f, Std=%.4f' % (np.mean(recall_res), np.std(recall_res)))
        print('Precision: Mean=%.4f, Std=%.4f' % (np.mean(precision_res), np.std(precision_res)))
        # print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)' % (np.mean(acc_res), np.std(acc_res), np.mean(f1_res), np.std(f1_res)))
        # predict(features, adj, data['Sample'].tolist(), data.index.tolist())
        # Extract disease name from featuredata path
        match = re.search(r'latent_(\w+)\.csv$', args.featuredata)
        if match:
            disease_name = match.group(1)
        else:
            disease_name = "Noname"

        # Create a dictionary to save parameters and results
        results = {
            'mode': [args.mode],
            'seed': [args.seed],
            'featuredata': [args.featuredata],
            'phylogeneTreedata': [args.phylogeneTreedata],
            'adjdata': [args.adjdata],
            'labeldata': [args.labeldata],
            'testsample': [args.testsample],
            'device': [args.device],
            'epochs': [args.epochs],
            'learningrate': [args.learningrate],
            'weight_decay': [args.weight_decay],
            'hidden': [args.hidden],
            'dropout': [args.dropout],
            'threshold': [args.threshold],
            'nclass': [args.nclass],
            'patience': [args.patience],
            'accuracy_mean': [np.mean(acc_res)],
            'accuracy_std': [np.std(acc_res)],
            'f1_score_mean': [np.mean(f1_res)],
            'f1_score_std': [np.std(f1_res)],
            'auc_mean': [np.mean(auc_res)],
            'auc_std': [np.std(auc_res)],
            'aupr_mean': [np.mean(aupr_res)],
            'aupr_std': [np.std(aupr_res)],
            'recall_mean': [np.mean(recall_res)],
            'recall_std': [np.std(recall_res)],
            'precision_mean': [np.mean(precision_res)],
            'precision_std': [np.std(precision_res)]
        }

        # Convert the results dictionary to a DataFrame
        results_df = pd.DataFrame(results)

        # Determine the mode for saving the file
        file_path = f'model/{disease_name}_GCN_results.csv'
        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)

        # Save results to CSV file with disease name in the filename
        results_df.to_csv(file_path, mode=mode, header=header, index=False)

    elif args.mode == 1:
        # load test samples
        test_sample_df = pd.read_csv(args.testsample, header=0, index_col=None)
        test_sample = test_sample_df.iloc[:, 0].tolist()
        all_sample = data['Sample'].tolist()
        train_sample = list(set(all_sample)-set(test_sample))

        #get index of train samples and test samples
        train_idx = data[data['Sample'].isin(train_sample)].index.tolist()
        test_idx = data[data['Sample'].isin(test_sample)].index.tolist()
        print(train_idx)
        print(test_idx)
        # GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
        # 使用DeepGCN搭建网络
        GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, n_blocks=4, dropout=args.dropout)
        GCN_model.to(device)
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx, dtype=torch.long, device=device)



        '''
        save a best model (with the minimum loss value)
        if the loss didn't decrease in N epochs，stop the train process.
        N can be set by args.patience 
        '''
        loss_values = []    #record the loss value of each epoch
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000   #record the lowest loss value
        for epoch in range(args.epochs):
            loss_values.append(train(epoch, optimizer, features, adj, labels, idx_train))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1     #In this epoch, the loss value didn't decrease

            if bad_counter == args.patience:
                break

            #save model of this epoch
            torch.save(GCN_model.state_dict(), 'model/GCN/{}.pkl'.format(epoch))

            #reserve the best model, delete other models
            files = glob.glob('model/GCN/*.pkl')
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                # print(file, name, epoch_nb)
                if epoch_nb != best_epoch:
                    os.remove(file)

        print('Training finished.')
        print('The best epoch model is ',best_epoch)
        GCN_model.load_state_dict(torch.load('model/GCN/{}.pkl'.format(best_epoch)))
        predict(features, adj, test_sample, idx_test,labels)

    print('Finished!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6} seconds")
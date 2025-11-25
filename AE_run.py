import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import autoencoder_model
import torch
import torch.utils.data as Data
import re
import os
import time
start_time = time.time()

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def work(data, in_feas, lr=0.001, bs=32, epochs=500, device=torch.device('cuda'), l=400,a=0.4, b=0.6, mode=0, topn=100):
    #name of sample
    sample_name = data['Sample'].tolist()

    #change data to a Tensor
    X,Y = data.iloc[:,1:].values, np.zeros(data.shape[0])   # X 选取Dataframe中所有的行和从第二列到最后一列 .value将选取的数据框转换为Numpy数组。Y全零数组。
    TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    #train a AE model
    if mode == 0 or mode == 1:
        print('Training model...')
        Tensor_data = Data.TensorDataset(TX, TY)
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)

        #initialize a model
        mmae = autoencoder_model.MMAE(in_feas, latent_dim=l, a=a, b=b)
        mmae.to(device)
        mmae.train()
        mmae.train_MMAE(train_loader, learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()       #before save and test, fix the variables
        torch.save(mmae, 'model/AE/MMAE_model.pkl')

    #load saved model, used for reducing dimensions
    if mode == 0 or mode == 2:
        print('Get the latent layer output...')
        mmae = torch.load('model/AE/MMAE_model.pkl')
        omics_1 = TX[:, :in_feas[0]]
        omics_2 = TX[:, in_feas[0]:in_feas[0]+in_feas[1]]

        # 使用加载的模型进行前向传播，得到潜在数据和解码后的各部分数据
        latent_data, decoded_omics_1, decoded_omics_2 = mmae.forward(omics_1, omics_2)
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        # 在Dataframe的第一列插入样本列
        latent_df.insert(0, 'Sample', sample_name)
        #save the integrated data(dim=100)
        # latent_df.to_csv('result/latent_WT2D.csv', header=True, index=False)
        latent_df.to_csv(f'result/latent_{disease_name}.csv', header=True, index=False)


    print('Extract features...')
    extract_features(data, in_feas, epochs, topn)
    return

def extract_features(data, in_feas, epochs, topn=100):
    # extract features
    #get each omics data
    data_omics_1 = data.iloc[:, 1: 1+in_feas[0]]
    data_omics_2 = data.iloc[:, 1+in_feas[0]: 1+in_feas[0]+in_feas[1]]

    #get all features of each omics data
    feas_omics_1 = data_omics_1.columns.tolist()
    feas_omics_2 = data_omics_2.columns.tolist()

    #calculate the standard deviation of each feature
    std_omics_1 = data_omics_1.std(axis=0)
    std_omics_2 = data_omics_2.std(axis=0)

    #record top N features every 10 epochs
    topn_omics_1 = pd.DataFrame()
    topn_omics_2 = pd.DataFrame()

    #used for feature extraction, epoch_ls = [10,20,...], if epochs % 10 != 0, add the last epoch
    epoch_ls = list(range(10, epochs+10,10))
    if epochs %10 != 0:
        epoch_ls.append(epochs)
    for epoch in tqdm(epoch_ls):
        #load model
        mmae = torch.load('model/AE/model_{}.pkl'.format(epoch))
        #get model variables
        model_dict = mmae.state_dict()

        #get the  absolutevalue of weights, the shape of matrix is (n_features, latent_layer_dim)
        weight_omics1 = np.abs(model_dict['encoder_omics_1.0.weight'].detach().cpu().numpy().T)
        weight_omics2 = np.abs(model_dict['encoder_omics_2.0.weight'].detach().cpu().numpy().T)

        weight_omics1_df = pd.DataFrame(weight_omics1, index=feas_omics_1)
        weight_omics2_df = pd.DataFrame(weight_omics2, index=feas_omics_2)

        #calculate the weight sum of each feature --> sum of each row
        weight_omics1_df['Weight_sum'] = weight_omics1_df.apply(lambda x:x.sum(), axis=1)
        weight_omics2_df['Weight_sum'] = weight_omics2_df.apply(lambda x:x.sum(), axis=1)

        weight_omics1_df['Std'] = std_omics_1
        weight_omics2_df['Std'] = std_omics_2

        #importance = Weight * Std
        weight_omics1_df['Importance'] = weight_omics1_df['Weight_sum']*weight_omics1_df['Std']
        weight_omics2_df['Importance'] = weight_omics2_df['Weight_sum']*weight_omics2_df['Std']

        #select top N features
        fea_omics_1_top = weight_omics1_df.nlargest(topn, 'Importance').index.tolist()
        fea_omics_2_top = weight_omics2_df.nlargest(topn, 'Importance').index.tolist()

        #save top N features in a dataframe
        col_name = 'epoch_'+str(epoch)
        topn_omics_1[col_name] = fea_omics_1_top
        topn_omics_2[col_name] = fea_omics_2_top

    #all of top N features
    # topn_omics_1.to_csv('result/marker_WT2D_1.csv', header=True, index=False)
    # topn_omics_2.to_csv('result/speciese_WT2D_2.csv', header=True, index=False)
    topn_omics_1.to_csv(f'result/marker_{disease_name}_1.csv', header=True, index=False)
    topn_omics_2.to_csv(f'result/speciese_{disease_name}_2.csv', header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, choices=[0,1,2], default=0,
                        help='Mode 0: train&intagrate, Mode 1: just train, Mode 2: just intagrate, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=101, help='Random seed, default=0.')
    parser.add_argument('--path1', '-p1', type=str, required=True, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name. ')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='Training batchszie, default: 32.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.0008, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=600, help='Training epochs, default: 500.')
    parser.add_argument('--latent', '-l', type=int, default=400, help='The latent layer dim, default: 400.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu', help='Training on cpu or gpu, default: gpu.')
    parser.add_argument('--a', '-a', type=float, default=0.4, help='[0,1], float, weight for the first omics data')
    parser.add_argument('--b', '-b', type=float, default=0.6, help='[0,1], float, weight for the second omics data.')
    parser.add_argument('--topn', '-n', type=int, default=100, help='Extract top N features every 10 epochs, default: 100.')
    args = parser.parse_args()

    #read data
    omics_data1 = pd.read_csv(args.path1, header=0, index_col=None)
    omics_data2 = pd.read_csv(args.path2, header=0, index_col=None)

    #Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #set random seed
    setup_seed(args.seed)

    if args.a + args.b != 1.0:
        print('The sum of weights must be 1.')
        exit(1)

    #dims of each omics data
    in_feas = [omics_data1.shape[1] - 1, omics_data2.shape[1] - 1]
    omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)

    omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
    print(omics_data1.shape)
    print(omics_data2.shape)
    #merge the multi-omics data, calculate on common samples
    Merge_data = pd.merge(omics_data1, omics_data2, on='Sample', how='inner')
    Merge_data.sort_values(by='Sample', ascending=True, inplace=True)
    # 创建一个字典来保存参数
    match = re.search(r'_(\w+)\.csv$', args.path1)
    if match:
        disease_name = match.group(1)
    else:
        disease_name = "Noname"
    #train model, reduce dimensions and extract features
    work(Merge_data, in_feas, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, device=device, l=args.latent,a=args.a, b=args.b, mode=args.mode, topn=args.topn)

    # Create a dictionary to save parameters
    params = {
        'mode': [args.mode],
        'seed': [args.seed],
        'path1': [args.path1],
        'path2': [args.path2],
        'batchsize': [args.batchsize],
        'learningrate': [args.learningrate],
        'epoch': [args.epoch],
        'latent': [args.latent],
        'device': [args.device],
        'a': [args.a],
        'b': [args.b],
        'topn': [args.topn]
    }

    params_df = pd.DataFrame(params)

    # Determine the mode for saving the file
    file_path = f'model/{disease_name}_AE_parameters.csv'
    mode = 'a' if os.path.exists(file_path) else 'w'
    header = not os.path.exists(file_path)

    # Save parameters to CSV file with disease name in the filename
    params_df.to_csv(file_path, mode=mode, header=header, index=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6} seconds")
    print('Success! Save in result dic')

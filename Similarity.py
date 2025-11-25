import snf
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='Location of input file')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='braycurtis',
                        help='Distance metric to compute.')
    args = parser.parse_args()

    print('Load data file...')
    omics_data = pd.read_csv(args.file, header=0, index_col=None)
    print(omics_data.shape)

    omics_data.rename(columns={omics_data.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data.sort_values(by='Sample', ascending=True, inplace=True)

    print('Compute similarity matrix...')
    # Compute pairwise distances and then convert to a similarity matrix (1-distance)
    distance_matrix = pdist(omics_data.iloc[:, 1:].values.astype(float), metric=args.metric)
    similarity_matrix = 1 - squareform(distance_matrix)

    print('Save similarity matrix...')
    similarity_df = pd.DataFrame(similarity_matrix, index=omics_data['Sample'], columns=omics_data['Sample'])
    similarity_df.to_csv('result/similarity_matrix_cosine_IBD_A.csv', header=True, index=True)

    # Optional: Create a clustermap
    fig = sns.clustermap(similarity_df, cmap='vlag', figsize=(8, 8))
    fig.savefig('result/similarity_clustermap_cosine_IBD_A.png', dpi=300)

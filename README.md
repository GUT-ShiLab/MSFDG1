# MSFDG

### 1. Data

The datasets utilized in this study comprise processed microbial relative abundance data and strain-level marker profiles from “***Machine Learning Meta-analysis of Large Metagenomic Datasets: Tools and Biological Insights*** ” by Pasolli. Specifically, for the six datasets, Pasolli et al. meticulously reprocessed the raw data according to the Standard Operating Procedures (SOP) of the Human Microbiome Project. To ensure the quality of the sequencing reads, those shorter than 90 nucleotides were excluded. Samples failing to meet these criteria were discarded from subsequent analyses. The MetaPhlAn2 software, with default settings, was then used to extract species-level relative abundances and strain-level marker profiles from the preprocessed metagenomic samples. All data are available for direct download at https://github.com/SegataLab/metaml/tree/master/data.

### 2. Code

（1）Environment Configuration

|   Package    | Version |
| :----------: | :-----: |
|    numpy     | 1.26.4  |
|    pandas    |  2.2.2  |
|    python    | 3.9.19  |
|   pytorch    |  2.4.0  |
| pytorch-cuda |  11.8   |

（2）Code Example

First, execute the `Similarity.py` script to generate six similarity matrices for the specific disease dataset based on different strategies and fuse them using the SNF algorithm.

```python
python Similarity.py
```

Next, run the `AE_run.py` script to obtain the latent representations (`latent_Cirrhosis.csv`) of the multi-input-output autoencoder for the specific disease dataset.

```python
python AE_run.py -p1 ./data/abundance_Cirrhosis.csv -p2 ./data/marker_Cirrhosis.csv -e 600 -l 400 -a 0.4 -b 0.6 -bs 32 -lr 0.0008
```

Finally, execute the `GCN_run.py` script to perform host disease prediction.

```python
python GCN_run.py -ph ./data/phylogenTree_p_Cirrhosis.csv -fd ./result/latent_Cirrhosis.csv -ad ./Similarity/fused_Cirrhosis_matrix.csv -ld ./data/labels_Cirrhosis.csv -lr 0.0001 -w 0.001 -hd 64 -e 500
```


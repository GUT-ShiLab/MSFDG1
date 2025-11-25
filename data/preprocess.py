import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_csv("phylogenTree_p_Obesity.csv", index_col=0)

# --------- 1. log 转换 ---------
# 加一个小常数防止 log(0)
log_df = np.log(df + 1e-6)
log_df.to_csv("phylogenTree_p_Obesitylog.csv")

# --------- 2. CLR 转换 ---------
def clr_transform(df, pseudocount=1e-6):
    """
    Centered Log-Ratio (CLR) transform
    对每一行进行处理：log(x / g(x))，其中 g(x) 为几何均值
    """
    # 替换0值以避免 log(0)
    df = df + pseudocount
    log_df = np.log(df)
    geometric_mean = log_df.mean(axis=1)
    clr_df = log_df.subtract(geometric_mean, axis=0)
    return clr_df

clr_df = clr_transform(df)
clr_df.to_csv("phylogenTree_p_Obesityclr.csv")

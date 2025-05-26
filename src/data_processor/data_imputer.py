import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def simple_imputation(df, method='mean'):
    """
    简单插补方法：使用均值、中位数或众数填充缺失值
    
    Args:
        df: 包含缺失值的DataFrame
        method: 插补方法，可选'mean', 'median', 'most_frequent', 'constant'
    
    Returns:
        填充后的DataFrame
    """
    imputer = SimpleImputer(strategy=method)
    # 只对数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df_numeric)
    return df_imputed

def knn_imputation(df, n_neighbors=5):
    """
    KNN插补方法：使用K近邻的平均值填充缺失值
    
    Args:
        df: 包含缺失值的DataFrame
        n_neighbors: 近邻数量
    
    Returns:
        填充后的DataFrame
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # 只对数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df_numeric)
    return df_imputed

def mice_imputation(df):
    """
    多重插补法：使用MICE(多重插补链方程)填充缺失值
    
    Args:
        df: 包含缺失值的DataFrame
    
    Returns:
        填充后的DataFrame
    """
    imputer = IterativeImputer(random_state=42, max_iter=10)
    # 只对数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df_numeric)
    return df_imputed

def forward_fill_imputation(df):
    """
    前向填充：使用时间序列中前一个值填充缺失值
    
    Args:
        df: 包含缺失值的DataFrame，需要按时间排序
    
    Returns:
        填充后的DataFrame
    """
    # 假设DataFrame已经按时间排序
    return df.ffill()

def feature_selection(df, missing_threshold=0.5, variance_threshold=0.01):
    """
    特征选择：移除缺失率高或方差低的特征
    
    Args:
        df: 包含特征的DataFrame
        missing_threshold: 缺失率阈值，高于此值的特征将被移除
        variance_threshold: 方差阈值，低于此值的特征将被移除
    
    Returns:
        筛选后的DataFrame和被移除的特征列表
    """
    # 计算每列的缺失率
    missing_rate = df.isnull().mean()
    
    # 移除缺失率高的特征
    high_missing_cols = missing_rate[missing_rate > missing_threshold].index.tolist()
    
    # 计算方差
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    
    # 移除方差低的特征
    low_var_cols = variances[variances < variance_threshold].index.tolist()
    
    # 合并要移除的特征
    cols_to_remove = list(set(high_missing_cols + low_var_cols))
    
    # 筛选特征
    selected_df = df.drop(columns=cols_to_remove)
    
    return selected_df, cols_to_remove

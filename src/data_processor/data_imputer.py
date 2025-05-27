import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

logger = logging.getLogger(__name__)

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
    
    # 记录缺失值情况
    total_missing = df_numeric.isnull().sum().sum()
    if total_missing > 0:
        logger.info(f"使用{method}方法插补{total_missing}个缺失值")
        
        # 执行插补
        df_imputed[numeric_cols] = imputer.fit_transform(df_numeric)
        
        # 检查是否仍有缺失值
        remaining_missing = df_imputed[numeric_cols].isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"插补后仍有{remaining_missing}个缺失值，使用0填充")
            df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(0)
    
    # 处理无穷大值
    inf_mask = np.isinf(df_imputed[numeric_cols])
    inf_count = inf_mask.sum().sum()
    if inf_count > 0:
        logger.warning(f"检测到{inf_count}个无穷大值，替换为有限值")
        for col in numeric_cols:
            col_median = df_numeric[col].median()
            col_std = df_numeric[col].std()
            # 将正无穷替换为中位数+3倍标准差，负无穷替换为中位数-3倍标准差
            df_imputed[col] = df_imputed[col].replace(np.inf, col_median + 3*col_std if pd.notnull(col_std) else col_median + 10)
            df_imputed[col] = df_imputed[col].replace(-np.inf, col_median - 3*col_std if pd.notnull(col_std) else col_median - 10)
    
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
    # 先使用简单插补处理极端缺失的列，避免KNN算法失效
    # 对于缺失率超过50%的列，先用均值填充
    missing_rates = df.isnull().mean()
    high_missing_cols = missing_rates[missing_rates > 0.5].index.tolist()
    
    if high_missing_cols:
        logger.warning(f"以下列的缺失率超过50%，先进行简单插补: {high_missing_cols}")
        df = simple_imputation(df, method='mean')
    
    # 使用KNN进行插补
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # 只对数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_imputed = df.copy()
    
    # 记录缺失值情况
    total_missing = df_numeric.isnull().sum().sum()
    if total_missing > 0:
        logger.info(f"使用KNN(k={n_neighbors})插补{total_missing}个缺失值")
        
        try:
            # 执行插补
            imputed_values = imputer.fit_transform(df_numeric)
            df_imputed[numeric_cols] = imputed_values
            
            # 检查是否仍有缺失值
            remaining_missing = df_imputed[numeric_cols].isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"KNN插补后仍有{remaining_missing}个缺失值，使用均值填充")
                # 使用均值填充剩余缺失值
                df_imputed = simple_imputation(df_imputed, method='mean')
        except Exception as e:
            logger.error(f"KNN插补失败: {e}，回退到简单插补")
            df_imputed = simple_imputation(df, method='mean')
    
    # 处理无穷大值
    inf_mask = np.isinf(df_imputed[numeric_cols])
    inf_count = inf_mask.sum().sum()
    if inf_count > 0:
        logger.warning(f"检测到{inf_count}个无穷大值，替换为有限值")
        for col in numeric_cols:
            col_median = df_numeric[col].median()
            col_std = df_numeric[col].std()
            # 将正无穷替换为中位数+3倍标准差，负无穷替换为中位数-3倍标准差
            df_imputed[col] = df_imputed[col].replace(np.inf, col_median + 3*col_std if pd.notnull(col_std) else col_median + 10)
            df_imputed[col] = df_imputed[col].replace(-np.inf, col_median - 3*col_std if pd.notnull(col_std) else col_median - 10)
    
    return df_imputed

def mice_imputation(df):
    """
    多重插补法：使用MICE(多重插补链方程)填充缺失值
    
    Args:
        df: 包含缺失值的DataFrame
    
    Returns:
        填充后的DataFrame
    """
    # 先使用简单插补处理极端缺失的列，避免MICE算法失效
    # 对于缺失率超过50%的列，先用均值填充
    missing_rates = df.isnull().mean()
    high_missing_cols = missing_rates[missing_rates > 0.5].index.tolist()
    
    if high_missing_cols:
        logger.warning(f"以下列的缺失率超过50%，先进行简单插补: {high_missing_cols}")
        df = simple_imputation(df, method='mean')
    
    # 使用MICE进行插补
    imputer = IterativeImputer(random_state=42, max_iter=10)
    # 只对数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_imputed = df.copy()
    
    # 记录缺失值情况
    total_missing = df_numeric.isnull().sum().sum()
    if total_missing > 0:
        logger.info(f"使用MICE方法插补{total_missing}个缺失值")
        
        try:
            # 执行插补
            imputed_values = imputer.fit_transform(df_numeric)
            df_imputed[numeric_cols] = imputed_values
            
            # 检查是否仍有缺失值
            remaining_missing = df_imputed[numeric_cols].isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"MICE插补后仍有{remaining_missing}个缺失值，使用均值填充")
                # 使用均值填充剩余缺失值
                df_imputed = simple_imputation(df_imputed, method='mean')
        except Exception as e:
            logger.error(f"MICE插补失败: {e}，回退到简单插补")
            df_imputed = simple_imputation(df, method='mean')
    
    # 处理无穷大值
    inf_mask = np.isinf(df_imputed[numeric_cols])
    inf_count = inf_mask.sum().sum()
    if inf_count > 0:
        logger.warning(f"检测到{inf_count}个无穷大值，替换为有限值")
        for col in numeric_cols:
            col_median = df_numeric[col].median()
            col_std = df_numeric[col].std()
            # 将正无穷替换为中位数+3倍标准差，负无穷替换为中位数-3倍标准差
            df_imputed[col] = df_imputed[col].replace(np.inf, col_median + 3*col_std if pd.notnull(col_std) else col_median + 10)
            df_imputed[col] = df_imputed[col].replace(-np.inf, col_median - 3*col_std if pd.notnull(col_std) else col_median - 10)
    
    return df_imputed

def forward_fill_imputation(df):
    """
    前向填充：使用时间序列中前一个值填充缺失值，对于第一个时间点的缺失值使用后向填充或均值
    
    Args:
        df: 包含缺失值的DataFrame，需要按时间排序
    
    Returns:
        填充后的DataFrame
    """
    # 假设DataFrame已经按时间排序
    df_imputed = df.copy()
    
    # 记录缺失值情况
    total_missing = df_imputed.isnull().sum().sum()
    if total_missing > 0:
        logger.info(f"使用前向填充方法处理{total_missing}个缺失值")
        
        # 前向填充
        df_imputed = df_imputed.ffill()
        
        # 检查是否仍有缺失值（通常是时间序列起始处的缺失）
        remaining_missing = df_imputed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.info(f"前向填充后仍有{remaining_missing}个缺失值（可能是序列起始处），尝试后向填充")
            
            # 后向填充
            df_imputed = df_imputed.bfill()
            
            # 检查是否仍有缺失值（如果整列都是NaN）
            still_missing = df_imputed.isnull().sum().sum()
            if still_missing > 0:
                logger.warning(f"前向和后向填充后仍有{still_missing}个缺失值，使用均值填充")
                # 使用均值填充剩余缺失值
                df_imputed = simple_imputation(df_imputed, method='mean')
    
    # 处理无穷大值
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df_imputed[numeric_cols])
    inf_count = inf_mask.sum().sum()
    if inf_count > 0:
        logger.warning(f"检测到{inf_count}个无穷大值，替换为有限值")
        for col in numeric_cols:
            col_median = df[col].median()
            col_std = df[col].std()
            # 将正无穷替换为中位数+3倍标准差，负无穷替换为中位数-3倍标准差
            df_imputed[col] = df_imputed[col].replace(np.inf, col_median + 3*col_std if pd.notnull(col_std) else col_median + 10)
            df_imputed[col] = df_imputed[col].replace(-np.inf, col_median - 3*col_std if pd.notnull(col_std) else col_median - 10)
    
    return df_imputed

def hybrid_imputation(df, time_col=None):
    """
    混合插补方法：结合多种插补策略，优先使用时间相关插补，然后是KNN和均值
    
    Args:
        df: 包含缺失值的DataFrame
        time_col: 时间列名，如果提供则会按时间排序再进行前向填充
    
    Returns:
        填充后的DataFrame
    """
    # 首先检查缺失值情况
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        return df
    
    logger.info(f"使用混合插补方法处理{total_missing}个缺失值")
    
    # 复制数据
    df_imputed = df.copy()
    
    # 如果提供了时间列，按时间排序
    if time_col is not None and time_col in df.columns:
        df_imputed = df_imputed.sort_values(by=time_col)
    
    # 1. 对时间序列数据使用前向填充
    df_imputed = forward_fill_imputation(df_imputed)
    
    # 2. 对仍有缺失的值使用KNN插补
    remaining_missing = df_imputed.isnull().sum().sum()
    if remaining_missing > 0:
        logger.info(f"前向填充后仍有{remaining_missing}个缺失值，使用KNN插补")
        
        # 只对数值列进行KNN插补
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            try:
                # 对数值列使用KNN插补
                numeric_df = df_imputed[numeric_cols]
                # 只对缺失率低于50%的列使用KNN
                missing_rates = numeric_df.isnull().mean()
                knn_cols = missing_rates[missing_rates < 0.5].index.tolist()
                
                if knn_cols:
                    # 应用KNN插补
                    imputer = KNNImputer(n_neighbors=min(5, len(df_imputed)-1))
                    df_imputed[knn_cols] = imputer.fit_transform(numeric_df[knn_cols])
            except Exception as e:
                logger.warning(f"KNN插补失败: {e}")
    
    # 3. 最后使用简单插补处理任何剩余的缺失值
    still_missing = df_imputed.isnull().sum().sum()
    if still_missing > 0:
        logger.info(f"KNN插补后仍有{still_missing}个缺失值，使用简单插补")
        df_imputed = simple_imputation(df_imputed, method='mean')
    
    # 处理无穷大值
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df_imputed[numeric_cols])
    inf_count = inf_mask.sum().sum()
    if inf_count > 0:
        logger.warning(f"检测到{inf_count}个无穷大值，替换为有限值")
        for col in numeric_cols:
            col_median = df_imputed[col].median()
            col_std = df_imputed[col].std()
            # 将正无穷替换为中位数+3倍标准差，负无穷替换为中位数-3倍标准差
            df_imputed[col] = df_imputed[col].replace(np.inf, col_median + 3*col_std if pd.notnull(col_std) else col_median + 10)
            df_imputed[col] = df_imputed[col].replace(-np.inf, col_median - 3*col_std if pd.notnull(col_std) else col_median - 10)
    
    return df_imputed

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

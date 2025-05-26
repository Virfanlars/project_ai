#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据插补脚本，提供多种缺失值处理方法
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processor.data_imputer import (
    simple_imputation, 
    knn_imputation, 
    mice_imputation, 
    forward_fill_imputation,
    feature_selection
)
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据插补工具')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='输出目录路径')
    
    # 插补方法
    parser.add_argument('--method', type=str, default='simple',
                        choices=['simple', 'knn', 'mice', 'forward_fill', 'feature_selection'],
                        help='插补方法')
    
    # 各方法特定参数
    parser.add_argument('--strategy', type=str, default='mean',
                        choices=['mean', 'median', 'most_frequent', 'constant'],
                        help='简单插补策略（用于simple方法）')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='K近邻数量（用于knn方法）')
    parser.add_argument('--missing_threshold', type=float, default=0.5,
                        help='缺失率阈值（用于feature_selection方法）')
    parser.add_argument('--variance_threshold', type=float, default=0.01,
                        help='方差阈值（用于feature_selection方法）')
    
    # 其他参数
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"imputed_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        setup_logger(args.log_level, os.path.join(output_dir, 'imputation.log'))
        logger = logging.getLogger(__name__)
        logger.info(f"启动数据插补工具，参数: {vars(args)}")
        
        # 检查数据目录
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"数据目录 {args.data_dir} 不存在")
        
        # 加载数据
        logger.info("加载数据...")
        
        # 尝试加载患者基本信息
        patient_file = os.path.join(args.data_dir, 'all_patients.csv')
        if os.path.exists(patient_file):
            patients_df = pd.read_csv(patient_file)
            logger.info(f"加载了患者数据: {len(patients_df)}行")
        else:
            logger.warning(f"患者数据文件不存在: {patient_file}")
            patients_df = None
        
        # 尝试加载生命体征数据
        vitals_file = os.path.join(args.data_dir, 'vital_signs_interim_5.csv')
        if os.path.exists(vitals_file):
            vitals_df = pd.read_csv(vitals_file)
            logger.info(f"加载了生命体征数据: {len(vitals_df)}行")
        else:
            logger.warning(f"生命体征数据文件不存在: {vitals_file}")
            vitals_df = None
        
        # 尝试加载实验室值数据
        labs_file = os.path.join(args.data_dir, 'lab_values.csv')
        if os.path.exists(labs_file):
            labs_df = pd.read_csv(labs_file)
            logger.info(f"加载了实验室值数据: {len(labs_df)}行")
        else:
            logger.warning(f"实验室值数据文件不存在: {labs_file}")
            labs_df = None
        
        # 尝试加载时间序列数据
        timeseries_file = os.path.join(args.data_dir, 'time_series.csv')
        if os.path.exists(timeseries_file):
            timeseries_df = pd.read_csv(timeseries_file)
            logger.info(f"加载了时间序列数据: {len(timeseries_df)}行")
        else:
            logger.warning(f"时间序列数据文件不存在: {timeseries_file}")
            timeseries_df = None
        
        # 进行缺失值分析
        logger.info("分析缺失值情况...")
        
        dfs_to_process = []
        df_names = []
        
        if patients_df is not None:
            dfs_to_process.append(patients_df)
            df_names.append('patients')
        
        if vitals_df is not None:
            dfs_to_process.append(vitals_df)
            df_names.append('vitals')
        
        if labs_df is not None:
            dfs_to_process.append(labs_df)
            df_names.append('labs')
        
        if timeseries_df is not None:
            dfs_to_process.append(timeseries_df)
            df_names.append('timeseries')
        
        # 保存缺失值分析结果
        missing_analysis = {}
        for i, df in enumerate(dfs_to_process):
            df_name = df_names[i]
            missing_rates = df.isnull().mean().sort_values(ascending=False)
            missing_analysis[df_name] = missing_rates.to_dict()
            
            # 保存缺失率分析
            missing_rates_df = pd.DataFrame({'feature': missing_rates.index, 'missing_rate': missing_rates.values})
            missing_rates_df.to_csv(os.path.join(output_dir, f"{df_name}_missing_rates.csv"), index=False)
            
            # 记录高缺失率特征
            high_missing = missing_rates[missing_rates > 0.5]
            if not high_missing.empty:
                logger.warning(f"{df_name}数据中有{len(high_missing)}个特征的缺失率超过50%: {', '.join(high_missing.index.tolist())}")
        
        # 执行插补
        logger.info(f"执行{args.method}插补方法...")
        
        processed_dfs = []
        
        for i, df in enumerate(dfs_to_process):
            df_name = df_names[i]
            logger.info(f"处理{df_name}数据...")
            
            # 根据选定的方法执行插补
            if args.method == 'simple':
                processed_df = simple_imputation(df, method=args.strategy)
                logger.info(f"使用{args.strategy}策略进行简单插补")
            
            elif args.method == 'knn':
                processed_df = knn_imputation(df, n_neighbors=args.n_neighbors)
                logger.info(f"使用K={args.n_neighbors}的KNN插补")
            
            elif args.method == 'mice':
                processed_df = mice_imputation(df)
                logger.info("使用MICE多重插补")
            
            elif args.method == 'forward_fill':
                processed_df = forward_fill_imputation(df)
                logger.info("使用时间序列前向填充")
            
            elif args.method == 'feature_selection':
                processed_df, removed_cols = feature_selection(
                    df, 
                    missing_threshold=args.missing_threshold,
                    variance_threshold=args.variance_threshold
                )
                logger.info(f"执行特征选择，移除了{len(removed_cols)}个特征: {', '.join(removed_cols)}")
            
            # 保存处理后的数据
            processed_df.to_csv(os.path.join(output_dir, f"{df_name}_imputed.csv"), index=False)
            logger.info(f"保存处理后的{df_name}数据，行数: {len(processed_df)}，列数: {len(processed_df.columns)}")
            
            processed_dfs.append(processed_df)
        
        # 分析插补效果
        logger.info("分析插补效果...")
        
        for i, (original_df, processed_df) in enumerate(zip(dfs_to_process, processed_dfs)):
            df_name = df_names[i]
            
            # 计算原始缺失值总数和比例
            original_missing = original_df.isnull().sum().sum()
            original_total = original_df.size
            original_missing_rate = original_missing / original_total * 100
            
            # 计算处理后缺失值总数和比例
            processed_missing = processed_df.isnull().sum().sum()
            processed_total = processed_df.size
            processed_missing_rate = processed_missing / processed_total * 100
            
            logger.info(f"{df_name}数据插补前: 缺失值{original_missing}/{original_total} ({original_missing_rate:.2f}%)")
            logger.info(f"{df_name}数据插补后: 缺失值{processed_missing}/{processed_total} ({processed_missing_rate:.2f}%)")
        
        logger.info(f"数据插补完成，结果保存在 {output_dir}")
        
    except Exception as e:
        logger.error(f"数据插补过程中发生错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
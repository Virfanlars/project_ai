#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症数据处理工具
此脚本将提取的原始数据转换为可用于模型训练的格式
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 命令行参数解析
parser = argparse.ArgumentParser(description='处理脓毒症数据')
parser.add_argument('--max_patients', type=int, default=None, help='最大处理患者数量')
parser.add_argument('--sample_patients', type=int, default=5000, help='要处理的样本患者数量')
args = parser.parse_args()

# 确保日志目录存在
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "data_processing.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 设置数据目录
DATA_DIR = os.path.join(project_root, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 生命体征和实验室检查的映射关系
VITALS_MAP = {
    220045: "心率",
    220050: "收缩压",
    220051: "舒张压",
    220052: "平均动脉压",
    220210: "呼吸频率",
    223761: "体温",
    220277: "血氧饱和度"
}

LABS_MAP = {
    50912: "乳酸",
    50971: "白细胞计数",
    51006: "血尿素氮",
    51265: "血小板计数",
    50983: "钠",
    50822: "氯化物",
    50813: "肌酐"
}

def load_patient_data():
    """
    加载患者基本信息
    
    返回:
        患者信息DataFrame
    """
    logger.info("加载患者基本信息...")
    
    patient_file = os.path.join(DATA_DIR, "patients_info.csv")
    
    if not os.path.exists(patient_file):
        logger.error(f"患者信息文件 {patient_file} 不存在!")
        return None
    
    df = pd.read_csv(patient_file)
    
    # 限制患者数量
    if args.max_patients is not None and len(df) > args.max_patients:
        logger.info(f"限制患者数量为 {args.max_patients} (原始数量: {len(df)})")
        df = df.head(args.max_patients)
    
    logger.info(f"加载了 {len(df)} 名患者的基本信息")
    
    # 转换日期时间列
    for col in ['intime', 'outtime', 'sepsis_onset_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

def load_vitals_data():
    """
    加载生命体征数据
    
    返回:
        生命体征DataFrame
    """
    logger.info("加载生命体征数据...")
    
    vitals_file = os.path.join(DATA_DIR, "vital_signs.csv")
    
    if not os.path.exists(vitals_file):
        logger.error(f"生命体征数据文件 {vitals_file} 不存在!")
        return None
    
    df = pd.read_csv(vitals_file)
    logger.info(f"加载了 {len(df)} 条生命体征记录")
    
    # 转换日期时间列
    if 'charttime' in df.columns:
        df['charttime'] = pd.to_datetime(df['charttime'])
    
    # 映射itemid到名称
    df['measurement'] = df['itemid'].map(VITALS_MAP)
    
    return df

def load_labs_data():
    """
    加载实验室检查数据
    
    返回:
        实验室检查DataFrame
    """
    logger.info("加载实验室检查数据...")
    
    labs_file = os.path.join(DATA_DIR, "lab_values.csv")
    
    if not os.path.exists(labs_file):
        logger.error(f"实验室检查数据文件 {labs_file} 不存在!")
        return None
    
    df = pd.read_csv(labs_file)
    logger.info(f"加载了 {len(df)} 条实验室检查记录")
    
    # 转换日期时间列
    if 'charttime' in df.columns:
        df['charttime'] = pd.to_datetime(df['charttime'])
    
    # 映射itemid到名称
    df['measurement'] = df['itemid'].map(LABS_MAP)
    
    return df

def create_time_series(patient_data, vitals_data, labs_data):
    """
    创建时间序列数据
    
    参数:
        patient_data: 患者信息DataFrame
        vitals_data: 生命体征DataFrame
        labs_data: 实验室检查DataFrame
        
    返回:
        时间序列DataFrame
    """
    logger.info("创建时间序列数据...")
    
    # 获取患者列表
    patients = patient_data['subject_id'].unique()
    logger.info(f"处理 {len(patients)} 名患者的时间序列数据")
    
    # 初始化结果列表
    results = []
    
    # 每小时的时间窗口
    window_size = 1  # 小时
    
    # 处理每个患者
    for i, subject_id in enumerate(patients):
        if i % 100 == 0:
            logger.info(f"处理第 {i+1}/{len(patients)} 名患者")
        
        # 获取该患者的信息
        patient_info = patient_data[patient_data['subject_id'] == subject_id].iloc[0]
        
        # 获取入院和出院时间
        intime = patient_info['intime']
        outtime = patient_info['outtime']
        
        # 获取脓毒症发作时间（如果有）
        sepsis_onset_time = patient_info.get('sepsis_onset_time')
        sepsis_label = 1 if not pd.isna(sepsis_onset_time) else 0
        
        # 获取该患者的生命体征和实验室检查数据
        patient_vitals = vitals_data[vitals_data['subject_id'] == subject_id]
        patient_labs = labs_data[labs_data['subject_id'] == subject_id]
        
        # 为每个小时时间窗口创建记录
        current_time = intime
        while current_time < outtime:
            window_end = current_time + timedelta(hours=window_size)
            
            # 当前时间窗口内的数据
            window_vitals = patient_vitals[(patient_vitals['charttime'] >= current_time) & 
                                          (patient_vitals['charttime'] < window_end)]
            window_labs = patient_labs[(patient_labs['charttime'] >= current_time) & 
                                      (patient_labs['charttime'] < window_end)]
            
            # 计算每个度量的平均值
            vitals_values = {}
            for item_id, name in VITALS_MAP.items():
                values = window_vitals[window_vitals['itemid'] == item_id]['valuenum']
                vitals_values[name] = values.mean() if not values.empty else np.nan
            
            labs_values = {}
            for item_id, name in LABS_MAP.items():
                values = window_labs[window_labs['itemid'] == item_id]['valuenum']
                labs_values[name] = values.mean() if not values.empty else np.nan
            
            # 计算距离入院的小时数
            hours_since_admission = (current_time - intime).total_seconds() / 3600
            
            # 创建记录
            record = {
                'subject_id': subject_id,
                'stay_id': patient_info['stay_id'],
                'timestamp': current_time,
                'hours_since_admission': hours_since_admission,
                'sepsis_label': sepsis_label
            }
            
            # 如果有脓毒症，计算距离发作的小时数（可能为负，表示还未发作）
            if sepsis_label == 1:
                hours_to_sepsis = (sepsis_onset_time - current_time).total_seconds() / 3600
                record['hours_to_sepsis'] = hours_to_sepsis
                
                # 标记当前时间窗口是否在脓毒症发作前12小时内
                record['sepsis_prediction_window'] = 1 if 0 <= hours_to_sepsis <= 12 else 0
            else:
                record['hours_to_sepsis'] = np.nan
                record['sepsis_prediction_window'] = 0
            
            # 添加生命体征和实验室检查值
            record.update(vitals_values)
            record.update(labs_values)
            
            # 添加到结果列表
            results.append(record)
            
            # 移动到下一个时间窗口
            current_time = window_end
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    logger.info(f"创建了 {len(df)} 条时间序列记录")
    
    return df

def create_ml_dataset(time_series):
    """
    创建用于机器学习的数据集
    
    参数:
        time_series: 时间序列DataFrame
        
    返回:
        机器学习数据集DataFrame
    """
    logger.info("创建机器学习数据集...")
    
    # 复制数据
    df = time_series.copy()
    
    # 添加滞后特征（前几个小时的值）
    lag_hours = [1, 2, 4, 8, 12, 24]  # 滞后小时数
    
    # 获取要创建滞后特征的列
    feature_columns = list(VITALS_MAP.values()) + list(LABS_MAP.values())
    
    # 对每个患者分组处理
    grouped = df.groupby(['subject_id', 'stay_id'])
    result_dfs = []
    
    for _, group in grouped:
        # 排序按时间
        group = group.sort_values('timestamp')
        
        # 为每个特征创建滞后值
        for col in feature_columns:
            if col in group.columns:
                for lag in lag_hours:
                    group[f"{col}_lag{lag}h"] = group[col].shift(lag)
        
        result_dfs.append(group)
    
    # 合并结果
    result_df = pd.concat(result_dfs)
    
    # 丢弃前24小时的数据（因为它们缺少滞后特征）
    result_df = result_df[result_df['hours_since_admission'] >= 24]
    
    logger.info(f"创建了包含 {len(result_df)} 条记录和 {len(result_df.columns)} 个特征的数据集")
    
    return result_df

def split_dataset(df):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        df: 数据集DataFrame
        
    返回:
        训练集、验证集和测试集DataFrame
    """
    logger.info("分割数据集...")
    
    # 按患者分割，确保同一患者的所有记录都在同一个集合中
    patients = df['subject_id'].unique()
    np.random.shuffle(patients)
    
    # 70%训练，15%验证，15%测试
    train_size = int(len(patients) * 0.7)
    val_size = int(len(patients) * 0.15)
    
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size + val_size]
    test_patients = patients[train_size + val_size:]
    
    # 分割数据集
    train_df = df[df['subject_id'].isin(train_patients)]
    val_df = df[df['subject_id'].isin(val_patients)]
    test_df = df[df['subject_id'].isin(test_patients)]
    
    logger.info(f"训练集: {len(train_df)} 记录, {len(train_patients)} 名患者")
    logger.info(f"验证集: {len(val_df)} 记录, {len(val_patients)} 名患者")
    logger.info(f"测试集: {len(test_df)} 记录, {len(test_patients)} 名患者")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, time_series_df):
    """
    保存处理后的数据
    
    参数:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        test_df: 测试集DataFrame
        time_series_df: 时间序列DataFrame
    """
    logger.info("保存处理后的数据...")
    
    # 保存数据集
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    # 保存时间序列数据
    time_series_df.to_csv(os.path.join(PROCESSED_DIR, "time_series.csv"), index=False)
    
    logger.info(f"数据已保存到 {PROCESSED_DIR}")

def main():
    """
    主函数：处理脓毒症数据
    """
    start_time = time.time()
    logger.info(f"开始数据处理流程，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数设置: 最大患者数量={args.max_patients}, 样本患者数量={args.sample_patients}")
    
    try:
        # 加载数据
        patients = load_patient_data()
        vitals = load_vitals_data()
        labs = load_labs_data()
        
        if patients is None or vitals is None or labs is None:
            logger.error("数据加载失败，退出")
            return 1
        
        # 如果患者数量很多，随机采样指定数量的患者
        if args.sample_patients is not None and len(patients) > args.sample_patients:
            logger.info(f"从 {len(patients)} 名患者中随机采样 {args.sample_patients} 名患者")
            # 设置随机种子以确保可重复性
            np.random.seed(42)
            patient_ids = np.random.choice(patients['subject_id'].unique(), 
                                        size=args.sample_patients, 
                                        replace=False)
            patients = patients[patients['subject_id'].isin(patient_ids)]
            logger.info(f"采样后患者数量: {len(patients)}")
        
        # 创建时间序列数据
        time_series = create_time_series(patients, vitals, labs)
        
        # 创建机器学习数据集
        ml_dataset = create_ml_dataset(time_series)
        
        # 分割数据集
        train_df, val_df, test_df = split_dataset(ml_dataset)
        
        # 保存处理后的数据
        save_processed_data(train_df, val_df, test_df, time_series)
        
        # 计算处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"数据处理完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        return 0  # 成功完成
    
    except Exception as e:
        logger.error(f"数据处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1  # 失败

if __name__ == "__main__":
    sys.exit(main()) 
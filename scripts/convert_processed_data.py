#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据格式转换脚本
将process_sepsis_data.py生成的数据转换为model_training.py需要的格式
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 确保日志目录存在
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "data_conversion.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 设置数据目录
PROCESSED_DIR = os.path.join(project_root, "data", "processed")
OUTPUT_DIR = PROCESSED_DIR

# 生命体征和实验室值映射
VITALS_MAP = {
    "心率": "heart_rate",
    "收缩压": "systolic_bp",
    "舒张压": "diastolic_bp",
    "平均动脉压": "map",
    "呼吸频率": "respiratory_rate",
    "体温": "temperature",
    "血氧饱和度": "spo2"
}

LABS_MAP = {
    "乳酸": "lactate",
    "白细胞计数": "wbc",
    "血尿素氮": "bun",
    "血小板计数": "platelet",
    "钠": "sodium",
    "氯化物": "chloride",
    "肌酐": "creatinine"
}

# 抗生素和血管活性药物列(模拟值)
ANTIBIOTICS = ["antibiotic_1", "antibiotic_2", "antibiotic_3", "antibiotic_4", "antibiotic_5"]
VASOPRESSORS = ["vasopressor_1", "vasopressor_2", "vasopressor_3", "vasopressor_4"]

def load_processed_data():
    """加载处理好的数据"""
    logger.info("加载处理好的数据...")
    
    train_file = os.path.join(PROCESSED_DIR, "train.csv")
    val_file = os.path.join(PROCESSED_DIR, "val.csv")
    test_file = os.path.join(PROCESSED_DIR, "test.csv")
    
    # 加载数据集
    if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        logger.info(f"加载了 {len(train_df)} 条训练记录，{len(val_df)} 条验证记录，{len(test_df)} 条测试记录")
        return train_df, val_df, test_df
    else:
        logger.error("找不到处理好的数据文件！请先运行 process_sepsis_data.py")
        return None, None, None

def transform_data(df):
    """转换数据格式"""
    logger.info("转换数据格式...")
    
    # 1. 创建患者特征DataFrame
    patient_features = df.copy()
    
    # 2. 重命名列，匹配期望的格式
    renamed_cols = {}
    
    # 转换生命体征名称
    for old_name, new_name in VITALS_MAP.items():
        if old_name in patient_features.columns:
            renamed_cols[old_name] = new_name
    
    # 转换实验室值名称
    for old_name, new_name in LABS_MAP.items():
        if old_name in patient_features.columns:
            renamed_cols[old_name] = new_name
    
    # 应用重命名
    if renamed_cols:
        patient_features = patient_features.rename(columns=renamed_cols)
    
    # 3. 添加模拟的抗生素和血管活性药物列
    for col in ANTIBIOTICS + VASOPRESSORS:
        if col not in patient_features.columns:
            # 使用随机值模拟药物使用（大多数时间为0）
            patient_features[col] = np.random.random(len(patient_features)) < 0.05
            patient_features[col] = patient_features[col].astype(int)
    
    # 4. 添加简单的文本嵌入列
    embed_dim = 32
    text_embed_cols = [f'text_embed_{i}' for i in range(embed_dim)]
    
    for col in text_embed_cols:
        if col not in patient_features.columns:
            # 使用随机值作为文本嵌入
            patient_features[col] = np.random.normal(0, 0.1, len(patient_features))
    
    # 5. 创建脓毒症标签DataFrame
    sepsis_labels = df[['subject_id', 'stay_id', 'timestamp', 'hours_since_admission', 
                        'sepsis_label', 'sepsis_prediction_window']].copy()
    
    # 添加SOFA评分（用sepsis_prediction_window * 随机值来模拟）
    sepsis_labels['sofa_score'] = np.where(
        sepsis_labels['sepsis_prediction_window'] == 1,
        np.random.randint(4, 12, len(sepsis_labels)),  # 脓毒症患者的高SOFA评分
        np.random.randint(0, 3, len(sepsis_labels))    # 非脓毒症患者的低SOFA评分
    )
    
    # 6. 重命名timestamp列为hour
    sepsis_labels['hour'] = sepsis_labels['hours_since_admission'].astype(int)
    
    return patient_features, sepsis_labels

def create_kg_embeddings(patient_features):
    """创建知识图谱嵌入"""
    logger.info("创建知识图谱嵌入...")
    
    # 生成简单的知识图谱嵌入
    n_concepts = 1000  # 概念数量
    embed_dim = 64     # 嵌入维度
    
    # 随机生成嵌入向量
    kg_embeddings = np.random.normal(0, 0.1, size=(n_concepts, embed_dim))
    
    return kg_embeddings

def create_time_axis():
    """创建时间轴信息"""
    logger.info("创建时间轴信息...")
    
    # 生成简单的时间轴信息
    reference_time = datetime.now()
    time_axis = {
        "resolution": "1H",
        "min_time": (reference_time - pd.Timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
        "max_time": reference_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return time_axis

def save_converted_data(patient_features, sepsis_labels, kg_embeddings, time_axis):
    """保存转换后的数据"""
    logger.info("保存转换后的数据...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存患者特征
    output_file = os.path.join(OUTPUT_DIR, "patient_features.csv")
    patient_features.to_csv(output_file, index=False)
    logger.info(f"患者特征已保存到: {output_file}")
    
    # 保存脓毒症标签
    output_file = os.path.join(OUTPUT_DIR, "sepsis_labels.csv")
    sepsis_labels.to_csv(output_file, index=False)
    logger.info(f"脓毒症标签已保存到: {output_file}")
    
    # 保存知识图谱嵌入
    output_file = os.path.join(OUTPUT_DIR, "kg_embeddings.npy")
    np.save(output_file, kg_embeddings)
    logger.info(f"知识图谱嵌入已保存到: {output_file}")
    
    # 保存时间轴信息
    output_file = os.path.join(OUTPUT_DIR, "time_axis.json")
    with open(output_file, 'w') as f:
        json.dump(time_axis, f)
    logger.info(f"时间轴信息已保存到: {output_file}")

def main():
    """主函数"""
    start_time = time.time()
    logger.info("开始数据格式转换...")
    
    # 1. 加载处理好的数据
    train_df, val_df, test_df = load_processed_data()
    if train_df is None:
        return 1
    
    # 2. 合并所有数据集进行处理
    all_data = pd.concat([train_df, val_df, test_df])
    logger.info(f"合并后共有 {len(all_data)} 条记录")
    
    # 3. 转换数据格式
    patient_features, sepsis_labels = transform_data(all_data)
    
    # 4. 创建知识图谱嵌入
    kg_embeddings = create_kg_embeddings(patient_features)
    
    # 5. 创建时间轴信息
    time_axis = create_time_axis()
    
    # 6. 保存转换后的数据
    save_converted_data(patient_features, sepsis_labels, kg_embeddings, time_axis)
    
    # 7. 完成
    elapsed_time = time.time() - start_time
    logger.info(f"数据格式转换完成，耗时: {elapsed_time:.2f} 秒")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
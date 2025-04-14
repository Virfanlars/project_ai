#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症早期预警系统 - 数据处理脚本
负责从MIMIC-IV数据库中提取并处理临床数据
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入模块
try:
    from utils.database_config import get_connection, DATABASE_CONFIG
    from scripts.data_extraction_main import main as extract_data
    from scripts.sepsis_labeling import label_sepsis_cases
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保您在项目根目录执行此脚本，并且已安装所有依赖")
    sys.exit(1)

def generate_sample_data():
    """生成样本数据用于测试"""
    logger.info("生成样本数据...")
    
    # 确保目录存在
    os.makedirs("data/processed", exist_ok=True)
    
    # 生成患者基本信息
    n_patients = 200
    patient_ids = [f"P{i:06d}" for i in range(1, n_patients+1)]
    
    # 随机分配脓毒症标签
    np.random.seed(42)
    sepsis_labels = np.random.binomial(1, 0.25, n_patients)  # 25%的患者为脓毒症
    
    # 创建患者信息DataFrame
    patient_info = pd.DataFrame({
        'subject_id': patient_ids,
        'hadm_id': [f"H{i:07d}" for i in range(1, n_patients+1)],
        'stay_id': [f"S{i:08d}" for i in range(1, n_patients+1)],
        'gender': np.random.choice(['M', 'F'], n_patients),
        'age': np.random.normal(65, 15, n_patients).astype(int),
        'sepsis_label': sepsis_labels,
        'sepsis_onset_time': [pd.Timestamp('2023-01-01') + pd.Timedelta(hours=np.random.randint(24, 72)) 
                              if label == 1 else None for label in sepsis_labels]
    })
    
    # 保存患者信息
    patient_info.to_csv("data/processed/patient_info.csv", index=False)
    logger.info(f"生成了 {n_patients} 名模拟患者信息")
    
    # 生成生命体征和实验室检查时序数据
    time_points = 36  # 每个患者的时间点数量
    vitals_columns = ['heart_rate', 'respiratory_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2']
    labs_columns = ['wbc', 'lactate', 'creatinine', 'platelet', 'bilirubin']
    
    # 为每个患者生成数据
    all_rows = []
    for i, patient_id in enumerate(patient_ids):
        hadm_id = f"H{i+1:07d}"
        stay_id = f"S{i+1:08d}"
        
        # 设置基线值
        vital_baselines = {
            'heart_rate': np.random.normal(85, 10),
            'respiratory_rate': np.random.normal(18, 3),
            'systolic_bp': np.random.normal(120, 15),
            'diastolic_bp': np.random.normal(80, 10),
            'temperature': np.random.normal(37, 0.5),
            'spo2': np.random.normal(96, 2)
        }
        
        lab_baselines = {
            'wbc': np.random.normal(8, 2),
            'lactate': np.random.normal(1.5, 0.5),
            'creatinine': np.random.normal(1.0, 0.3),
            'platelet': np.random.normal(250, 50),
            'bilirubin': np.random.normal(0.8, 0.2)
        }
        
        # 若患者有脓毒症，生命体征和实验室值会随时间恶化
        sepsis = sepsis_labels[i] == 1
        onset_time = None
        if sepsis:
            onset_time = np.random.randint(time_points // 3, time_points * 2 // 3)
        
        for t in range(time_points):
            charttime = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=t)
            
            # 创建基础行
            row = {
                'subject_id': patient_id,
                'hadm_id': hadm_id,
                'stay_id': stay_id,
                'charttime': charttime
            }
            
            # 添加模拟数据噪声
            for col in vitals_columns:
                baseline = vital_baselines[col]
                # 如果有脓毒症且超过发作时间，加入趋势
                if sepsis and t >= onset_time:
                    trend_effect = (t - onset_time) * 0.05
                    if col == 'heart_rate':
                        row[col] = baseline + baseline * trend_effect + np.random.normal(0, 3)
                    elif col == 'respiratory_rate':
                        row[col] = baseline + baseline * trend_effect + np.random.normal(0, 1)
                    elif col == 'systolic_bp':
                        row[col] = baseline - baseline * trend_effect + np.random.normal(0, 5)
                    elif col == 'diastolic_bp':
                        row[col] = baseline - baseline * trend_effect + np.random.normal(0, 3)
                    elif col == 'temperature':
                        row[col] = baseline + 0.1 * trend_effect + np.random.normal(0, 0.1)
                    elif col == 'spo2':
                        row[col] = baseline - baseline * 0.03 * trend_effect + np.random.normal(0, 1)
                else:
                    row[col] = baseline + np.random.normal(0, baseline * 0.05)
            
            # 实验室值
            for col in labs_columns:
                # 实验室检查不是每小时都有，随机生成
                if np.random.random() < 0.3:  # 30%概率有实验室检查
                    baseline = lab_baselines[col]
                    if sepsis and t >= onset_time:
                        trend_effect = (t - onset_time) * 0.08
                        if col == 'wbc':
                            row[col] = baseline + baseline * trend_effect + np.random.normal(0, 0.5)
                        elif col == 'lactate':
                            row[col] = baseline + baseline * trend_effect + np.random.normal(0, 0.2)
                        elif col == 'creatinine':
                            row[col] = baseline + baseline * 0.5 * trend_effect + np.random.normal(0, 0.1)
                        elif col == 'platelet':
                            row[col] = baseline - baseline * 0.3 * trend_effect + np.random.normal(0, 10)
                        elif col == 'bilirubin':
                            row[col] = baseline + baseline * 0.3 * trend_effect + np.random.normal(0, 0.1)
                    else:
                        row[col] = baseline + np.random.normal(0, baseline * 0.1)
                else:
                    # 缺失值
                    row[col] = np.nan
            
            # 添加药物特征（抗生素、血管活性药物）
            drug_columns = ['antibiotic_1', 'antibiotic_2', 'antibiotic_3', 'antibiotic_4', 'antibiotic_5',
                           'vasopressor_1', 'vasopressor_2', 'vasopressor_3', 'vasopressor_4']
            
            for col in drug_columns:
                # 如果有脓毒症且超过发作时间，有更高概率使用药物
                if sepsis and t >= onset_time:
                    prob = 0.1 + 0.3 * min(1, (t - onset_time) / 10)  # 随着时间增加概率
                    row[col] = 1 if np.random.random() < prob else 0
                else:
                    # 正常情况下低概率使用药物
                    row[col] = 1 if np.random.random() < 0.05 else 0
            
            all_rows.append(row)
    
    # 创建DataFrame并保存
    aligned_data = pd.DataFrame(all_rows)
    aligned_data.to_csv("data/processed/aligned_data.csv", index=False)
    
    # 保存脓毒症标签
    sepsis_data = patient_info[['subject_id', 'sepsis_label', 'sepsis_onset_time']]
    sepsis_data.to_csv("data/processed/sepsis_labels.csv", index=False)
    
    logger.info(f"生成了 {len(all_rows)} 条时序数据记录")
    logger.info(f"数据生成完成并保存到 data/processed/ 目录")
    
    return patient_info, aligned_data

def run_data_extraction():
    """从数据库中提取和处理数据"""
    logger.info("执行数据提取...")
    
    # 测试数据库连接
    try:
        conn = get_connection()
        logger.info("数据库连接成功")
        conn.close()
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        logger.warning("无法连接到数据库，将回退到使用样本数据")
        return False
    
    # 执行数据提取
    try:
        logger.info("开始执行数据提取主程序...")
        extract_data()
        logger.info("数据提取完成")
        return True
    except Exception as e:
        logger.error(f"数据提取过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_processed_data():
    """检查处理后的数据"""
    expected_files = [
        "data/processed/patient_info.csv",
        "data/processed/aligned_data.csv"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"以下预期文件不存在: {missing_files}")
        return False
    
    # 显示数据基本信息
    try:
        patient_info = pd.read_csv("data/processed/patient_info.csv")
        aligned_data = pd.read_csv("data/processed/aligned_data.csv")
        
        logger.info(f"患者信息: {len(patient_info)} 行, {patient_info.shape[1]} 列")
        logger.info(f"时序数据: {len(aligned_data)} 行, {aligned_data.shape[1]} 列")
        
        sepsis_count = patient_info['sepsis_label'].sum()
        sepsis_pct = (sepsis_count / len(patient_info)) * 100
        logger.info(f"脓毒症患者: {sepsis_count} ({sepsis_pct:.1f}%)")
        
        # 检查时序数据完整性
        subject_ids = patient_info['subject_id'].unique()
        aligned_subject_ids = aligned_data['subject_id'].unique()
        common_ids = set(subject_ids).intersection(set(aligned_subject_ids))
        
        logger.info(f"患者信息中的唯一ID数: {len(subject_ids)}")
        logger.info(f"时序数据中的唯一ID数: {len(aligned_subject_ids)}")
        logger.info(f"两者共有的ID数: {len(common_ids)}")
        
        return True
    except Exception as e:
        logger.error(f"检查数据时出错: {e}")
        return False

def main():
    """主函数"""
    start_time = time.time()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='脓毒症预警系统数据处理')
    parser.add_argument('--sample', action='store_true', 
                      help='使用模拟样本数据而不是连接数据库')
    args = parser.parse_args()
    
    logger.info("开始数据处理流程")
    logger.info(f"使用样本数据: {args.sample}")
    
    if os.environ.get('SEPSIS_USE_MOCK_DATA', '').lower() == 'true':
        logger.info("检测到环境变量SEPSIS_USE_MOCK_DATA=true，将使用模拟数据")
        args.sample = True
    
    # 判断是否使用样本数据
    if args.sample:
        logger.info("使用模拟样本数据...")
        generate_sample_data()
    else:
        logger.info("从MIMIC-IV数据库提取数据...")
        success = run_data_extraction()
        if not success:
            logger.warning("数据库提取失败，尝试使用模拟样本数据...")
            generate_sample_data()
    
    # 检查生成的脓毒症标签
    if not os.path.exists("data/processed/sepsis_labels.csv"):
        logger.info("生成脓毒症标签...")
        try:
            label_sepsis_cases()
        except Exception as e:
            logger.error(f"脓毒症标签生成失败: {e}")
    
    # 检查处理后的数据
    logger.info("检查处理后的数据...")
    data_ok = check_processed_data()
    
    # 完成
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if data_ok:
        logger.info(f"数据处理完成，耗时: {elapsed_time:.2f} 秒")
        logger.info("数据已保存到 data/processed/ 目录")
    else:
        logger.error("数据处理未成功完成")
        sys.exit(1)

if __name__ == "__main__":
    main() 
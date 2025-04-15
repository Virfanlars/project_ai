#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症数据抽取工具
此脚本专注于从sepsis3表提取脓毒症患者数据，使用硬编码的数据库连接，避免编码问题
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import psycopg2
import traceback
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

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
        logging.FileHandler(os.path.join(log_dir, "data_extraction.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 硬编码数据库连接参数
DB_CONFIG = {
    'host': '172.16.3.67',
    'port': 5432,
    'dbname': 'mimiciv',
    'user': 'postgres',
    'password': '123456'
}

# 设置并确保数据目录存在
DATA_DIR = os.path.join(project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def get_connection_string():
    """
    获取数据库连接字符串
    
    返回:
        PostgreSQL连接字符串
    """
    return f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"

def optimize_db_connection(conn):
    """
    优化数据库连接设置
    
    参数:
        conn: 数据库连接对象
    """
    try:
        logger.info("优化数据库连接设置...")
        cursor = conn.cursor()
        
        # 优化参数
        optimization_params = {
            "work_mem": "256MB",
            "maintenance_work_mem": "512MB",
            "effective_cache_size": "2GB"
        }
        
        for param, value in optimization_params.items():
            try:
                cursor.execute(f"SET {param} = '{value}'")
                logger.debug(f"设置 {param} = {value}")
            except Exception as e:
                logger.warning(f"设置 {param} = {value} 失败: {e}")
        
        cursor.close()
        logger.info("数据库连接优化完成")
    except Exception as e:
        logger.warning(f"优化数据库连接失败: {e}")

def get_all_icu_patients(conn, min_stay_hours=24):
    """
    获取所有符合条件的ICU患者
    
    参数:
        conn: 数据库连接
        min_stay_hours: 最小ICU停留时间（小时）
        
    返回:
        DataFrame，包含患者ID和入院ID
    """
    logger.info(f"获取ICU患者数据 (最小停留时间: {min_stay_hours}小时)...")
    
    query = f"""
    SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id,
           i.intime, i.outtime,
           EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 AS stay_hours
    FROM mimiciv_icu.icustays i
    WHERE EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 >= {min_stay_hours}
    ORDER BY i.subject_id, i.hadm_id, i.stay_id
    """
    
    df = pd.read_sql(query, conn)
    logger.info(f"找到 {len(df)} 名符合条件的ICU患者")
    return df

def get_sepsis_patients(conn):
    """
    从MIMIC-IV数据库中检索脓毒症患者数据。
    
    参数:
        conn: 数据库连接对象
    
    返回:
        包含脓毒症患者信息的DataFrame
    """
    try:
        logger.info("获取脓毒症患者数据...")
        
        # 查询脓毒症患者数据
        query = """
        SELECT  s.subject_id,
                s.hadm_id,
                s.icustay_id as stay_id,
                s.infection_time as sepsis_onset_time,
                s.sepsis3,
                s.sepsis3_onset_order
        FROM    mimiciv_derived.sepsis3 s
        WHERE   s.sepsis3 = 1
                AND s.sepsis3_onset_order = 1
        ORDER BY s.icustay_id
        """
        
        df = pd.read_sql(query, conn)
        logger.info(f"检索到 {len(df)} 名脓毒症患者")
        return df
    
    except Exception as e:
        logger.error(f"获取脓毒症患者数据时出错: {e}")
        traceback.print_exc()
        raise

def extract_vital_signs(conn, subject_ids, stay_ids):
    """
    提取生命体征数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        stay_ids: ICU停留ID列表
        
    返回:
        生命体征DataFrame
    """
    logger.info("提取生命体征数据...")
    
    # 限制提取数量，避免查询过大
    if len(subject_ids) > 1000:
        logger.warning(f"患者数量过多 ({len(subject_ids)}), 限制为前1000名...")
        subject_ids = subject_ids[:1000]
        stay_ids = stay_ids[:1000]
    
    # 使用占位符构建查询
    subject_placeholders = ','.join(['%s'] * len(subject_ids))
    stay_placeholders = ','.join(['%s'] * len(stay_ids))
    
    query = f"""
    SELECT  ce.subject_id,
            ce.stay_id,
            ce.charttime,
            ce.itemid,
            ce.valuenum
    FROM    mimiciv_icu.chartevents ce
    WHERE   ce.subject_id IN ({subject_placeholders})
            AND ce.stay_id IN ({stay_placeholders})
            AND ce.itemid IN (
                220045, -- 心率
                220050, -- 收缩压
                220051, -- 舒张压
                220052, -- 平均动脉压
                220210, -- 呼吸频率
                223761, -- 体温（摄氏度）
                220277  -- 血氧饱和度
            )
            AND ce.valuenum IS NOT NULL
    ORDER BY ce.subject_id, ce.stay_id, ce.charttime
    """
    
    # 组合参数列表
    params = subject_ids + stay_ids
    
    # 执行查询
    df = pd.read_sql_query(query, conn, params=params)
    
    logger.info(f"提取了 {len(df)} 条生命体征记录")
    return df

def extract_lab_values(conn, subject_ids, hadm_ids):
    """
    提取实验室检查数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        hadm_ids: 入院ID列表
        
    返回:
        实验室检查DataFrame
    """
    logger.info("提取实验室检查数据...")
    
    # 限制提取数量，避免查询过大
    if len(subject_ids) > 1000:
        logger.warning(f"患者数量过多 ({len(subject_ids)}), 限制为前1000名...")
        subject_ids = subject_ids[:1000]
        hadm_ids = hadm_ids[:1000]
    
    # 使用占位符构建查询
    subject_placeholders = ','.join(['%s'] * len(subject_ids))
    hadm_placeholders = ','.join(['%s'] * len(hadm_ids))
    
    query = f"""
    SELECT  le.subject_id,
            le.hadm_id,
            le.charttime,
            le.itemid,
            le.valuenum
    FROM    mimiciv_hosp.labevents le
    WHERE   le.subject_id IN ({subject_placeholders})
            AND le.hadm_id IN ({hadm_placeholders})
            AND le.itemid IN (
                50912, -- 乳酸
                50971, -- 白细胞计数
                51006, -- 血尿素氮
                51265, -- 血小板计数
                50983, -- 钠
                50822, -- 氯化物
                50813  -- 肌酐
            )
            AND le.valuenum IS NOT NULL
    ORDER BY le.subject_id, le.hadm_id, le.charttime
    """
    
    # 组合参数列表
    params = subject_ids + hadm_ids
    
    # 执行查询
    df = pd.read_sql_query(query, conn, params=params)
    
    logger.info(f"提取了 {len(df)} 条实验室检查记录")
    return df

def merge_patient_data(icu_patients, sepsis_patients):
    """
    合并患者数据，标记脓毒症患者
    
    参数:
        icu_patients: ICU患者DataFrame
        sepsis_patients: 脓毒症患者DataFrame
        
    返回:
        合并后的DataFrame
    """
    logger.info("合并ICU患者和脓毒症患者数据...")
    
    # 合并脓毒症信息到ICU患者表
    result = pd.merge(
        icu_patients, 
        sepsis_patients[['subject_id', 'hadm_id', 'stay_id', 'sepsis_onset_time']], 
        on=['subject_id', 'hadm_id', 'stay_id'], 
        how='left'
    )
    
    # 标记脓毒症患者
    result['sepsis_label'] = result['sepsis_onset_time'].notnull().astype(int)
    
    # 计算脓毒症比例
    sepsis_count = result['sepsis_label'].sum()
    total_count = len(result)
    sepsis_percent = (sepsis_count / total_count) * 100 if total_count > 0 else 0
    
    logger.info(f"合并后总患者数: {total_count}")
    logger.info(f"脓毒症患者数: {sepsis_count} ({sepsis_percent:.2f}%)")
    
    return result

def save_data(data, filename):
    """
    保存数据到CSV文件
    
    参数:
        data: 要保存的DataFrame
        filename: 文件名（不含路径）
    """
    filepath = os.path.join(DATA_DIR, filename)
    data.to_csv(filepath, index=False)
    logger.info(f"数据已保存到 {filepath}")

def main():
    """
    主函数：提取MIMIC-IV数据用于脓毒症预测
    """
    start_time = time.time()
    logger.info(f"开始数据提取流程，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 连接到数据库
        conn_string = get_connection_string()
        logger.info(f"连接到数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        conn = psycopg2.connect(conn_string)
        
        # 优化连接
        optimize_db_connection(conn)
        
        # 获取所有ICU患者
        icu_patients = get_all_icu_patients(conn, min_stay_hours=24)
        
        # 获取脓毒症患者
        sepsis_patients = get_sepsis_patients(conn)
        
        # 合并患者数据
        patients = merge_patient_data(icu_patients, sepsis_patients)
        
        # 保存患者基本信息
        save_data(patients, "patients_info.csv")
        
        # 限制提取的患者数量，避免过长的处理时间
        sample_size = min(len(patients), 1000)
        logger.info(f"从 {len(patients)} 名患者中采样 {sample_size} 名进行详细数据提取")
        
        sampled_patients = patients.sample(sample_size, random_state=42) if len(patients) > sample_size else patients
        
        # 保存采样的患者信息
        save_data(sampled_patients, "sampled_patients.csv")
        
        # 提取生命体征和实验室检查数据
        vitals = extract_vital_signs(conn, sampled_patients['subject_id'].tolist(), sampled_patients['stay_id'].tolist())
        labs = extract_lab_values(conn, sampled_patients['subject_id'].tolist(), sampled_patients['hadm_id'].tolist())
        
        # 保存提取的数据
        save_data(vitals, "vital_signs.csv")
        save_data(labs, "lab_values.csv")
        
        # 关闭连接
        conn.close()
        
        # 计算处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"数据提取完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        logger.info("数据提取摘要:")
        logger.info(f"- 总ICU患者数: {len(icu_patients)}")
        logger.info(f"- 脓毒症患者数: {len(sepsis_patients)}")
        logger.info(f"- 采样患者数: {len(sampled_patients)}")
        logger.info(f"- 提取的生命体征记录数: {len(vitals)}")
        logger.info(f"- 提取的实验室检查记录数: {len(labs)}")
        
        return 0  # 成功完成
    
    except Exception as e:
        logger.error(f"数据提取过程中出错: {e}")
        traceback.print_exc()
        return 1  # 失败

if __name__ == "__main__":
    sys.exit(main()) 
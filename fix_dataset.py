#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症数据提取与预处理工具
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import psycopg2
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data_extraction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 数据库连接参数
DB_CONFIG = {
    'host': '172.16.3.67',
    'port': 5432,
    'dbname': 'mimiciv',
    'user': 'postgres',
    'password': '123456'
}

# 设置数据目录
DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_db_connection():
    """获取数据库连接"""
    try:
        # 添加连接超时参数
        conn_string = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']} connect_timeout=30"
        conn = psycopg2.connect(conn_string)
        logger.info(f"成功连接到数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise

def optimize_db_connection(conn):
    """优化数据库连接设置"""
    try:
        cursor = conn.cursor()
        # 优化参数
        cursor.execute("SET work_mem = '256MB'")
        cursor.execute("SET maintenance_work_mem = '512MB'")
        cursor.execute("SET effective_cache_size = '2GB'")
        cursor.close()
        logger.info("数据库连接优化完成")
    except Exception as e:
        logger.warning(f"优化数据库连接失败: {e}")

def get_sepsis_patients(conn, limit=None):
    """
    从MIMIC-IV数据库获取脓毒症患者数据
    
    Args:
        conn: 数据库连接
        limit: 限制提取的患者数量
        
    Returns:
        DataFrame: 脓毒症患者数据
    """
    logger.info("获取脓毒症患者数据...")
    
    try:
        # 使用完整的字段列表
        query = """
        SELECT s.subject_id, s.hadm_id, s.icustay_id, 
               s.sepsis3, s.infection_time, s.infection_hour, s.hr,
               s.sofa_score, s.sofa_baseline, s.sofa_increase,
               s.sepsis3_onset_order,
               i.intime, i.outtime,
               EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 AS stay_hours
        FROM mimiciv_derived.sepsis3 s
        JOIN mimiciv_icu.icustays i ON s.icustay_id = i.stay_id
        WHERE s.sepsis3 = 1
        ORDER BY s.subject_id, s.hadm_id, s.icustay_id
        """
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        sepsis_df = pd.read_sql(query, conn)
        
        # 使用infection_time作为脓毒症发作时间
        sepsis_df['sepsis_onset_time'] = sepsis_df['infection_time']
        
        logger.info(f"找到 {len(sepsis_df)} 名脓毒症患者")
        
        # 保存脓毒症患者数据
        sepsis_file = os.path.join(PROCESSED_DIR, "sepsis_patients.csv")
        sepsis_df.to_csv(sepsis_file, index=False)
        logger.info(f"脓毒症患者数据已保存至 {sepsis_file}")
        
        return sepsis_df
    except Exception as e:
        logger.error(f"获取脓毒症患者数据失败: {e}")
        return pd.DataFrame()

def get_control_patients(conn, sepsis_count):
    """获取对照组患者"""
    logger.info("获取非脓毒症患者作为对照组...")
    
    try:
        # 查询非脓毒症患者作为对照组，使用与脓毒症患者相同的字段结构
        query = f"""
        SELECT i.subject_id, i.hadm_id, i.stay_id as icustay_id, 
               0 as sepsis3, NULL as infection_time, NULL as infection_hour, NULL as hr,
               NULL as sofa_score, NULL as sofa_baseline, NULL as sofa_increase,
               NULL as sepsis3_onset_order,
               i.intime, i.outtime,
               EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 AS stay_hours
        FROM mimiciv_icu.icustays i
        LEFT JOIN mimiciv_derived.sepsis3 s ON i.stay_id = s.icustay_id
        WHERE s.icustay_id IS NULL 
          AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 >= 24
        ORDER BY i.subject_id, i.hadm_id, i.stay_id
        LIMIT {sepsis_count * 2}
        """
        
        control_df = pd.read_sql(query, conn)
        
        # 添加sepsis_onset_time列，保持与脓毒症患者数据框结构一致
        control_df['sepsis_onset_time'] = None
        
        logger.info(f"找到 {len(control_df)} 名非脓毒症患者作为对照组")
        
        # 保存对照组患者数据
        control_file = os.path.join(PROCESSED_DIR, "control_patients.csv")
        control_df.to_csv(control_file, index=False)
        logger.info(f"对照组患者数据已保存至 {control_file}")
        
        return control_df
    except Exception as e:
        logger.error(f"获取对照组患者数据失败: {e}")
        return pd.DataFrame()

def extract_sofa_scores(conn, stay_ids, batch_size=100):
    """
    分批提取SOFA评分数据
    
    Args:
        conn: 数据库连接
        stay_ids: ICU停留ID列表
        batch_size: 每批处理的ID数量
        
    Returns:
        DataFrame: SOFA评分数据
    """
    logger.info(f"分批提取SOFA评分数据，共 {len(stay_ids)} 个stay_id，批次大小 {batch_size}")
    
    # 将ID列表分批
    batches = [stay_ids[i:i+batch_size] for i in range(0, len(stay_ids), batch_size)]
    logger.info(f"共分为 {len(batches)} 个批次")
    
    all_sofa = []
    
    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="提取SOFA评分数据")):
        # 构建IN子句
        ids_str = ','.join([f"'{id}'" for id in batch_ids])
        
        # 构建查询
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            icustay_id,
            hr,
            respiration_score,
            coagulation_score,
            liver_score,
            cardiovascular_score,
            sbp_score,
            cns_score,
            renal_score,
            vaso_score,
            sofa_score
        FROM 
            mimiciv_derived.sofa
        WHERE 
            icustay_id IN ({ids_str})
        ORDER BY 
            icustay_id, hr
        """
        
        try:
            batch_sofa = pd.read_sql(query, conn)
            
            if not batch_sofa.empty:
                all_sofa.append(batch_sofa)
                
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"已完成 {batch_idx + 1}/{len(batches)} 批次的SOFA评分数据提取")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx} 提取SOFA评分数据失败: {e}")
            continue
    
    if not all_sofa:
        logger.warning("未提取到任何SOFA评分数据！")
        return pd.DataFrame()
    
    sofa_df = pd.concat(all_sofa, ignore_index=True)
    logger.info(f"共提取了 {len(sofa_df)} 条SOFA评分记录")
    
    # 保存SOFA评分数据
    sofa_file = os.path.join(PROCESSED_DIR, "sofa_scores.csv")
    sofa_df.to_csv(sofa_file, index=False)
    logger.info(f"SOFA评分数据已保存至 {sofa_file}")
    
    return sofa_df

def extract_vital_signs(conn, stay_ids, batch_size=20, max_retry=3, timeout=300):
    """
    分批提取生命体征数据，包含超时处理和重试机制
    
    Args:
        conn: 数据库连接
        stay_ids: ICU停留ID列表
        batch_size: 每批处理的ID数量（减小为20）
        max_retry: 最大重试次数
        timeout: 查询超时时间(秒)
        
    Returns:
        DataFrame: 生命体征数据
    """
    logger.info(f"分批提取生命体征数据，共 {len(stay_ids)} 个stay_id，批次大小 {batch_size}")
    
    # 定义关键生命体征项目ID
    vital_signs_items = {
        220045: 'heart_rate',         # 心率
        220050: 'sbp',                # 收缩压
        220051: 'dbp',                # 舒张压
        220052: 'map',                # 平均动脉压
        220210: 'resp_rate',          # 呼吸频率
        223761: 'temperature',        # 体温（摄氏度）
        220277: 'spo2'                # 血氧饱和度
    }
    
    # 设置查询超时
    try:
        cursor = conn.cursor()
        cursor.execute(f"SET statement_timeout = {timeout * 1000}")  # 毫秒
        cursor.close()
        logger.info(f"设置查询超时: {timeout}秒")
    except Exception as e:
        logger.warning(f"设置查询超时失败: {e}")
    
    # 将ID列表分批，减小批次大小
    batches = [stay_ids[i:i+batch_size] for i in range(0, len(stay_ids), batch_size)]
    logger.info(f"共分为 {len(batches)} 个批次")
    
    all_vitals = []
    
    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="提取生命体征数据")):
        # 构建IN子句
        ids_str = ','.join([f"'{id}'" for id in batch_ids])
        
        # 添加更多限制条件减小结果集大小，只选取最近1年的数据
        query = f"""
        SELECT ce.subject_id, ce.stay_id, ce.charttime, ce.itemid, ce.valuenum
        FROM mimiciv_icu.chartevents ce
        WHERE ce.stay_id IN ({ids_str})
          AND ce.itemid IN ({','.join(map(str, vital_signs_items.keys()))})
          AND ce.valuenum IS NOT NULL
          AND ce.valuenum > 0  -- 排除无效值
          AND ce.valuenum < 1000  -- 排除异常值
        ORDER BY ce.stay_id, ce.charttime
        LIMIT 10000  -- 限制结果数量
        """
        
        # 实现重试机制
        retry_count = 0
        success = False
        
        while retry_count < max_retry and not success:
            try:
                # 创建新连接以避免长时间查询导致的连接问题
                new_conn = get_db_connection()
                batch_vitals = pd.read_sql(query, new_conn)
                new_conn.close()
                
                if not batch_vitals.empty:
                    # 添加特征名称列
                    batch_vitals['feature_name'] = batch_vitals['itemid'].map(vital_signs_items)
                    all_vitals.append(batch_vitals)
                
                success = True
                logger.info(f"批次 {batch_idx+1}/{len(batches)} 成功提取 {len(batch_vitals)} 条生命体征记录")
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"批次 {batch_idx} 提取失败 (尝试 {retry_count}/{max_retry}): {e}")
                time.sleep(5)  # 重试前等待5秒
                
        if not success:
            logger.error(f"批次 {batch_idx} 提取生命体征数据失败，已达最大重试次数")
            # 继续处理下一批，而不是整个中断
            continue
            
        # 每5个批次保存一次中间结果，避免全部丢失
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            if all_vitals:
                interim_df = pd.concat(all_vitals, ignore_index=True)
                interim_file = os.path.join(PROCESSED_DIR, f"vital_signs_interim_{batch_idx+1}.csv")
                interim_df.to_csv(interim_file, index=False)
                logger.info(f"已保存中间结果至 {interim_file}，当前共 {len(interim_df)} 条记录")
    
    if not all_vitals:
        logger.warning("未提取到任何生命体征数据！")
        return pd.DataFrame()
    
    try:
        vitals_df = pd.concat(all_vitals, ignore_index=True)
        logger.info(f"共提取了 {len(vitals_df)} 条生命体征记录")
        
        # 保存生命体征数据
        vitals_file = os.path.join(PROCESSED_DIR, "vital_signs.csv")
        vitals_df.to_csv(vitals_file, index=False)
        logger.info(f"生命体征数据已保存至 {vitals_file}")
        
        return vitals_df
    except Exception as e:
        logger.error(f"合并生命体征数据失败: {e}")
        
        # 尝试加载最新的中间结果
        latest_interim = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("vital_signs_interim_")])
        if latest_interim:
            latest_file = os.path.join(PROCESSED_DIR, latest_interim[-1])
            logger.info(f"加载最新的中间结果文件: {latest_file}")
            return pd.read_csv(latest_file)
        
        return pd.DataFrame()

def extract_lab_values(conn, hadm_ids, batch_size=20, max_retry=3, timeout=300):
    """
    分批提取实验室检查数据，包含超时处理和重试机制
    
    Args:
        conn: 数据库连接
        hadm_ids: 入院ID列表
        batch_size: 每批处理的ID数量
        max_retry: 最大重试次数
        timeout: 查询超时时间(秒)
        
    Returns:
        DataFrame: 实验室检查数据
    """
    logger.info(f"分批提取实验室检查数据，共 {len(hadm_ids)} 个hadm_id，批次大小 {batch_size}")
    
    # 定义关键实验室检查项目ID
    lab_items = {
        50912: 'lactate',          # 乳酸
        50971: 'wbc',              # 白细胞计数
        51006: 'bun',              # 血尿素氮
        51265: 'platelet',         # 血小板计数
        50983: 'sodium',           # 钠
        50822: 'chloride',         # 氯化物
        50813: 'creatinine',       # 肌酐
        50882: 'bilirubin_total'   # 总胆红素
    }
    
    # 设置查询超时
    try:
        cursor = conn.cursor()
        cursor.execute(f"SET statement_timeout = {timeout * 1000}")  # 毫秒
        cursor.close()
        logger.info(f"设置查询超时: {timeout}秒")
    except Exception as e:
        logger.warning(f"设置查询超时失败: {e}")
    
    # 将ID列表分批
    batches = [hadm_ids[i:i+batch_size] for i in range(0, len(hadm_ids), batch_size)]
    logger.info(f"共分为 {len(batches)} 个批次")
    
    all_labs = []
    
    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="提取实验室检查数据")):
        # 构建IN子句
        ids_str = ','.join([f"'{id}'" for id in batch_ids])
        
        # 构建查询，添加更多限制条件
        query = f"""
        SELECT le.subject_id, le.hadm_id, le.charttime, le.itemid, le.valuenum
        FROM mimiciv_hosp.labevents le
        WHERE le.hadm_id IN ({ids_str})
          AND le.itemid IN ({','.join(map(str, lab_items.keys()))})
          AND le.valuenum IS NOT NULL
          AND le.valuenum > 0  -- 排除无效值
          AND le.valuenum < 10000  -- 排除异常值
        ORDER BY le.hadm_id, le.charttime
        LIMIT 10000  -- 限制结果数量
        """
        
        # 实现重试机制
        retry_count = 0
        success = False
        
        while retry_count < max_retry and not success:
            try:
                # 创建新连接以避免长时间查询导致的连接问题
                new_conn = get_db_connection()
                batch_labs = pd.read_sql(query, new_conn)
                new_conn.close()
                
                if not batch_labs.empty:
                    # 添加特征名称列
                    batch_labs['feature_name'] = batch_labs['itemid'].map(lab_items)
                    all_labs.append(batch_labs)
                
                success = True
                logger.info(f"批次 {batch_idx+1}/{len(batches)} 成功提取 {len(batch_labs)} 条实验室检查记录")
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"批次 {batch_idx} 提取失败 (尝试 {retry_count}/{max_retry}): {e}")
                time.sleep(5)  # 重试前等待5秒
                
        if not success:
            logger.error(f"批次 {batch_idx} 提取实验室检查数据失败，已达最大重试次数")
            continue
            
        # 每5个批次保存一次中间结果
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            if all_labs:
                interim_df = pd.concat(all_labs, ignore_index=True)
                interim_file = os.path.join(PROCESSED_DIR, f"lab_values_interim_{batch_idx+1}.csv")
                interim_df.to_csv(interim_file, index=False)
                logger.info(f"已保存中间结果至 {interim_file}，当前共 {len(interim_df)} 条记录")
    
    if not all_labs:
        logger.warning("未提取到任何实验室检查数据！")
        return pd.DataFrame()
    
    try:
        labs_df = pd.concat(all_labs, ignore_index=True)
        logger.info(f"共提取了 {len(labs_df)} 条实验室检查记录")
        
        # 保存实验室检查数据
        labs_file = os.path.join(PROCESSED_DIR, "lab_values.csv")
        labs_df.to_csv(labs_file, index=False)
        logger.info(f"实验室检查数据已保存至 {labs_file}")
        
        return labs_df
    except Exception as e:
        logger.error(f"合并实验室检查数据失败: {e}")
        
        # 尝试加载最新的中间结果
        latest_interim = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("lab_values_interim_")])
        if latest_interim:
            latest_file = os.path.join(PROCESSED_DIR, latest_interim[-1])
            logger.info(f"加载最新的中间结果文件: {latest_file}")
            return pd.read_csv(latest_file)
        
        return pd.DataFrame()

def extract_medication_data(conn, hadm_ids, batch_size=100):
    """
    提取药物数据，主要关注抗生素和升压药
    
    Args:
        conn: 数据库连接
        hadm_ids: 入院ID列表
        batch_size: 每批处理的ID数量
        
    Returns:
        DataFrame: 药物数据
    """
    logger.info(f"分批提取药物数据，共 {len(hadm_ids)} 个hadm_id，批次大小 {batch_size}")
    
    # 将ID列表分批
    batches = [hadm_ids[i:i+batch_size] for i in range(0, len(hadm_ids), batch_size)]
    
    all_meds = []
    
    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="提取药物数据")):
        # 构建IN子句
        ids_str = ','.join([f"'{id}'" for id in batch_ids])
        
        # 抗生素查询 - 使用处方数据
        antibiotic_query = f"""
        SELECT 
            p.subject_id, 
            p.hadm_id, 
            p.starttime as charttime,
            'antibiotic' as drug_type,
            p.drug as drug_name
        FROM 
            mimiciv_hosp.prescriptions p
        WHERE 
            p.hadm_id IN ({ids_str})
            AND (
                p.drug ILIKE '%ceftriaxone%' OR
                p.drug ILIKE '%vancomycin%' OR
                p.drug ILIKE '%piperacillin%' OR
                p.drug ILIKE '%tazobactam%' OR
                p.drug ILIKE '%meropenem%' OR
                p.drug ILIKE '%ciprofloxacin%' OR
                p.drug ILIKE '%levofloxacin%' OR
                p.drug ILIKE '%metronidazole%'
            )
        """
        
        # 升压药查询 - 使用处方数据
        vasopressor_query = f"""
        SELECT 
            p.subject_id, 
            p.hadm_id, 
            p.starttime as charttime,
            'vasopressor' as drug_type,
            p.drug as drug_name
        FROM 
            mimiciv_hosp.prescriptions p
        WHERE 
            p.hadm_id IN ({ids_str})
            AND (
                p.drug ILIKE '%norepinephrine%' OR
                p.drug ILIKE '%epinephrine%' OR
                p.drug ILIKE '%vasopressin%' OR
                p.drug ILIKE '%phenylephrine%' OR
                p.drug ILIKE '%dopamine%'
            )
        """
        
        try:
            # 获取抗生素数据
            antibiotics_df = pd.read_sql(antibiotic_query, conn)
            
            # 获取升压药数据
            vasopressors_df = pd.read_sql(vasopressor_query, conn)
            
            # 合并两种药物数据
            if not antibiotics_df.empty or not vasopressors_df.empty:
                batch_meds = pd.concat([antibiotics_df, vasopressors_df], ignore_index=True)
                all_meds.append(batch_meds)
                
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"已完成 {batch_idx + 1}/{len(batches)} 批次的药物数据提取")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx} 提取药物数据失败: {e}")
            continue
    
    if not all_meds:
        logger.warning("未提取到任何药物数据！")
        return pd.DataFrame()
    
    meds_df = pd.concat(all_meds, ignore_index=True)
    logger.info(f"共提取了 {len(meds_df)} 条药物记录")
    
    # 保存药物数据
    meds_file = os.path.join(PROCESSED_DIR, "medications.csv")
    meds_df.to_csv(meds_file, index=False)
    logger.info(f"药物数据已保存至 {meds_file}")
    
    return meds_df

def process_time_series_data(patients_df, vitals_df, labs_df, sofa_df, meds_df, hours_before_sepsis=24, hours_after_sepsis=24, resample_freq='1H'):
    """
    处理时间序列数据，创建均匀间隔的时间序列
    
    Args:
        patients_df: 患者数据
        vitals_df: 生命体征数据
        labs_df: 实验室检查数据
        sofa_df: SOFA评分数据
        meds_df: 药物数据
        hours_before_sepsis: 脓毒症发作前的小时数
        hours_after_sepsis: 脓毒症发作后的小时数
        resample_freq: 重采样频率
        
    Returns:
        DataFrame: 处理后的时间序列数据
    """
    logger.info("处理时间序列数据...")
    
    # 创建时间序列数据框
    all_ts_data = []
    
    # 处理每个患者的数据
    for _, patient in tqdm(patients_df.iterrows(), total=len(patients_df), desc="处理患者数据"):
        subject_id = patient['subject_id']
        hadm_id = patient['hadm_id']
        icustay_id = patient['icustay_id']
        is_sepsis = patient['sepsis3'] == 1
        
        # 确定时间范围
        if is_sepsis and pd.notna(patient['sepsis_onset_time']):
            # 对于脓毒症患者，以脓毒症发作时间为中心
            sepsis_time = pd.to_datetime(patient['sepsis_onset_time'])
            start_time = sepsis_time - pd.Timedelta(hours=hours_before_sepsis)
            end_time = sepsis_time + pd.Timedelta(hours=hours_after_sepsis)
            
            # 确保不超出住院时间范围
            intime = pd.to_datetime(patient['intime'])
            outtime = pd.to_datetime(patient['outtime'])
            start_time = max(start_time, intime)
            end_time = min(end_time, outtime)
        else:
            # 对于非脓毒症患者，使用入院后的固定时间段
            intime = pd.to_datetime(patient['intime'])
            outtime = pd.to_datetime(patient['outtime'])
            
            # 确保住院时间足够长
            if (outtime - intime).total_seconds() / 3600 < (hours_before_sepsis + hours_after_sepsis):
                continue
                
            start_time = intime + pd.Timedelta(hours=24)  # 从入院24小时后开始
            end_time = start_time + pd.Timedelta(hours=hours_before_sepsis + hours_after_sepsis)
            
            # 确保不超出住院时间范围
            end_time = min(end_time, outtime)
        
        # 创建均匀时间间隔
        time_range = pd.date_range(start=start_time, end=end_time, freq=resample_freq)
        
        if len(time_range) < 12:  # 至少需要12个时间点
            logger.warning(f"患者 {subject_id} 的时间范围过短，跳过")
            continue
        
        # 为每个时间点创建记录
        for ts in time_range:
            # 计算与脓毒症发作时间的距离（小时）
            if is_sepsis and pd.notna(patient['sepsis_onset_time']):
                hours_to_sepsis = (sepsis_time - ts).total_seconds() / 3600
            else:
                hours_to_sepsis = None
            
            # 计算入院后小时数
            hours_since_admission = (ts - pd.to_datetime(patient['intime'])).total_seconds() / 3600
            
            # 创建基本记录
            record = {
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'icustay_id': icustay_id,
                'timestamp': ts,
                'hours_since_admission': hours_since_admission,
                'sepsis_label': 1 if is_sepsis else 0,
                'hours_to_sepsis': hours_to_sepsis,
                'sepsis_prediction_window': 1 if (is_sepsis and hours_to_sepsis > 0 and hours_to_sepsis <= 6) else 0
            }
            
            all_ts_data.append(record)
    
    # 创建时间序列数据框
    ts_df = pd.DataFrame(all_ts_data)
    
    if ts_df.empty:
        logger.error("未能创建任何时间序列数据！")
        return pd.DataFrame()
    
    logger.info(f"创建了 {len(ts_df)} 条时间序列记录")
    
    # 将生命体征数据添加到时间序列中
    if not vitals_df.empty:
        logger.info("添加生命体征数据...")
        
        # 确保时间戳列为datetime类型
        vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'])
        
        # 对每个生命体征特征进行处理
        vital_features = vitals_df['feature_name'].unique()
        
        for feature in tqdm(vital_features, desc="处理生命体征特征"):
            # 提取该特征的数据
            feature_data = vitals_df[vitals_df['feature_name'] == feature]
            
            # 为每个患者的每个时间点找到最近的测量值
            for idx, row in ts_df.iterrows():
                # 查找该患者的该特征数据
                patient_feature_data = feature_data[
                    (feature_data['subject_id'] == row['subject_id']) &
                    (feature_data['stay_id'] == row['icustay_id'])
                ]
                
                if patient_feature_data.empty:
                    continue
                    
                # 找到最接近时间点的测量
                ts = row['timestamp']
                patient_feature_data['time_diff'] = abs(patient_feature_data['charttime'] - ts)
                closest_idx = patient_feature_data['time_diff'].idxmin()
                closest_row = patient_feature_data.loc[closest_idx]
                
                # 如果时间差异在6小时内，使用该值
                if closest_row['time_diff'] <= pd.Timedelta(hours=6):
                    ts_df.at[idx, feature] = closest_row['valuenum']
    
    # 将实验室检查数据添加到时间序列中
    if not labs_df.empty:
        logger.info("添加实验室检查数据...")
        
        # 确保时间戳列为datetime类型
        labs_df['charttime'] = pd.to_datetime(labs_df['charttime'])
        
        # 对每个实验室检查特征进行处理
        lab_features = labs_df['feature_name'].unique()
        
        for feature in tqdm(lab_features, desc="处理实验室检查特征"):
            # 提取该特征的数据
            feature_data = labs_df[labs_df['feature_name'] == feature]
            
            # 为每个患者的每个时间点找到最近的测量值
            for idx, row in ts_df.iterrows():
                # 查找该患者的该特征数据
                patient_feature_data = feature_data[
                    (feature_data['subject_id'] == row['subject_id']) &
                    (feature_data['hadm_id'] == row['hadm_id'])
                ]
                
                if patient_feature_data.empty:
                    continue
                    
                # 找到最接近时间点的测量
                ts = row['timestamp']
                patient_feature_data['time_diff'] = abs(patient_feature_data['charttime'] - ts)
                closest_idx = patient_feature_data['time_diff'].idxmin()
                closest_row = patient_feature_data.loc[closest_idx]
                
                # 如果时间差异在12小时内，使用该值
                if closest_row['time_diff'] <= pd.Timedelta(hours=12):
                    ts_df.at[idx, feature] = closest_row['valuenum']
    
    # 将SOFA评分数据添加到时间序列中
    if not sofa_df.empty:
        logger.info("添加SOFA评分数据...")
        
        # 获取SOFA评分相关特征
        sofa_features = ['sofa_score', 'respiration_score', 'coagulation_score', 
                         'liver_score', 'cardiovascular_score', 'cns_score', 
                         'renal_score', 'vaso_score']
        
        # 为每个患者的每个时间点找到最接近的SOFA评分
        for idx, row in tqdm(ts_df.iterrows(), total=len(ts_df), desc="处理SOFA评分"):
            # 查找该患者的SOFA评分数据
            patient_sofa = sofa_df[sofa_df['icustay_id'] == row['icustay_id']]
            
            if patient_sofa.empty:
                continue
                
            # 找到最接近当前时间点的SOFA评分
            hours_since_adm = row['hours_since_admission']
            patient_sofa['hr_diff'] = abs(patient_sofa['hr'] - hours_since_adm)
            closest_idx = patient_sofa['hr_diff'].idxmin()
            closest_sofa = patient_sofa.loc[closest_idx]
            
            # 如果时间差异在24小时内，使用该SOFA评分
            if closest_sofa['hr_diff'] <= 24:
                for feature in sofa_features:
                    if feature in closest_sofa:
                        ts_df.at[idx, feature] = closest_sofa[feature]
    
    # 添加药物使用数据
    if not meds_df.empty:
        logger.info("添加药物使用数据...")
        
        # 确保时间戳列为datetime类型
        meds_df['charttime'] = pd.to_datetime(meds_df['charttime'])
        
        # 对抗生素和升压药分别进行处理
        for drug_type in ['antibiotic', 'vasopressor']:
            # 提取该类型的药物数据
            drug_data = meds_df[meds_df['drug_type'] == drug_type]
            
            if drug_data.empty:
                continue
                
            # 为每个患者的每个时间点检查是否使用了该类药物
            for idx, row in tqdm(ts_df.iterrows(), total=len(ts_df), desc=f"处理{drug_type}"):
                # 查找该患者的该类药物数据
                patient_drug_data = drug_data[
                    (drug_data['subject_id'] == row['subject_id']) &
                    (drug_data['hadm_id'] == row['hadm_id'])
                ]
                
                if patient_drug_data.empty:
                    continue
                    
                # 检查在该时间点前24小时内是否有该类药物使用
                ts = row['timestamp']
                recent_usage = patient_drug_data[
                    (patient_drug_data['charttime'] <= ts) &
                    (patient_drug_data['charttime'] >= ts - pd.Timedelta(hours=24))
                ]
                
                # 如果有使用，标记为1
                if not recent_usage.empty:
                    ts_df.at[idx, drug_type] = 1
                else:
                    ts_df.at[idx, drug_type] = 0
    
    # 保存时间序列数据
    ts_file = os.path.join(PROCESSED_DIR, "time_series.csv")
    ts_df.to_csv(ts_file, index=False)
    logger.info(f"时间序列数据已保存至 {ts_file}")
    
    return ts_df

def create_train_val_test_split(ts_df, test_size=0.15, val_size=0.15, random_state=42):
    """
    创建训练集、验证集和测试集
    
    Args:
        ts_df: 时间序列数据
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("创建训练集、验证集和测试集...")
    
    # 获取唯一患者ID
    unique_patients = ts_df[['subject_id', 'hadm_id', 'icustay_id']].drop_duplicates()
    
    # 分层抽样，确保脓毒症和非脓毒症患者的比例一致
    sepsis_flag = []
    for _, row in unique_patients.iterrows():
        # 检查该患者是否有脓毒症
        patient_data = ts_df[
            (ts_df['subject_id'] == row['subject_id']) &
            (ts_df['hadm_id'] == row['hadm_id']) &
            (ts_df['icustay_id'] == row['icustay_id'])
        ]
        has_sepsis = patient_data['sepsis_label'].max() == 1
        sepsis_flag.append(has_sepsis)
    
    unique_patients['has_sepsis'] = sepsis_flag
    
    # 分层拆分
    # 首先分离出测试集
    train_val_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=test_size,
        stratify=unique_patients['has_sepsis'],
        random_state=random_state
    )
    
    # 再从剩余数据中分离出验证集
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size / (1 - test_size),  # 调整比例
        stratify=train_val_patients['has_sepsis'],
        random_state=random_state
    )
    
    # 基于患者ID分割时间序列数据
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for _, row in train_patients.iterrows():
        patient_data = ts_df[
            (ts_df['subject_id'] == row['subject_id']) &
            (ts_df['hadm_id'] == row['hadm_id']) &
            (ts_df['icustay_id'] == row['icustay_id'])
        ]
        train_df = pd.concat([train_df, patient_data], ignore_index=True)
    
    for _, row in val_patients.iterrows():
        patient_data = ts_df[
            (ts_df['subject_id'] == row['subject_id']) &
            (ts_df['hadm_id'] == row['hadm_id']) &
            (ts_df['icustay_id'] == row['icustay_id'])
        ]
        val_df = pd.concat([val_df, patient_data], ignore_index=True)
    
    for _, row in test_patients.iterrows():
        patient_data = ts_df[
            (ts_df['subject_id'] == row['subject_id']) &
            (ts_df['hadm_id'] == row['hadm_id']) &
            (ts_df['icustay_id'] == row['icustay_id'])
        ]
        test_df = pd.concat([test_df, patient_data], ignore_index=True)
    
    logger.info(f"训练集: {len(train_df)} 行, {len(train_patients)} 名患者")
    logger.info(f"验证集: {len(val_df)} 行, {len(val_patients)} 名患者")
    logger.info(f"测试集: {len(test_df)} 行, {len(test_patients)} 名患者")
    
    # 统计各集合中脓毒症患者的比例
    train_sepsis = train_df['sepsis_label'].mean() * 100
    val_sepsis = val_df['sepsis_label'].mean() * 100
    test_sepsis = test_df['sepsis_label'].mean() * 100
    
    logger.info(f"训练集脓毒症患者比例: {train_sepsis:.2f}%")
    logger.info(f"验证集脓毒症患者比例: {val_sepsis:.2f}%")
    logger.info(f"测试集脓毒症患者比例: {test_sepsis:.2f}%")
    
    # 保存拆分后的数据
    train_file = os.path.join(PROCESSED_DIR, "train.csv")
    val_file = os.path.join(PROCESSED_DIR, "val.csv")
    test_file = os.path.join(PROCESSED_DIR, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    logger.info(f"训练集已保存至 {train_file}")
    logger.info(f"验证集已保存至 {val_file}")
    logger.info(f"测试集已保存至 {test_file}")
    
    return train_df, val_df, test_df

def main():
    """主函数：提取MIMIC-IV数据用于脓毒症预测"""
    start_time = time.time()
    logger.info(f"开始数据提取流程，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 数据量限制设置 - 由于设备限制，只处理少量数据用于实验
    DATA_LIMIT = True  # 设置为True启用数据限制
    MAX_PATIENTS = 100  # 每类患者最多处理的数量
    MAX_VITALS_PER_PATIENT = 100  # 每个患者最多提取的生命体征记录数
    MAX_LABS_PER_PATIENT = 50  # 每个患者最多提取的实验室检查记录数
    
    logger.info(f"数据限制模式: {'启用' if DATA_LIMIT else '禁用'}")
    if DATA_LIMIT:
        logger.info(f"每类患者最多处理: {MAX_PATIENTS}名")
        logger.info(f"每个患者最多生命体征记录: {MAX_VITALS_PER_PATIENT}条")
        logger.info(f"每个患者最多实验室检查记录: {MAX_LABS_PER_PATIENT}条")
    
    # 断点续传检查点目录
    checkpoint_dir = os.path.join(PROCESSED_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # 连接到数据库
        conn = get_db_connection()
        
        # 优化连接
        optimize_db_connection(conn)
        
        # 步骤1: 获取脓毒症患者
        sepsis_file = os.path.join(PROCESSED_DIR, "sepsis_patients.csv")
        if os.path.exists(sepsis_file):
            logger.info(f"加载已有的脓毒症患者数据: {sepsis_file}")
            sepsis_patients = pd.read_csv(sepsis_file)
            if DATA_LIMIT and len(sepsis_patients) > MAX_PATIENTS:
                logger.info(f"限制脓毒症患者数量为{MAX_PATIENTS}")
                sepsis_patients = sepsis_patients.drop_duplicates(subset=['subject_id']).head(MAX_PATIENTS)
        else:
            logger.info("提取脓毒症患者数据...")
            # 在查询中直接限制患者数量
            sepsis_limit = MAX_PATIENTS if DATA_LIMIT else 1000
            sepsis_patients = get_sepsis_patients(conn, limit=sepsis_limit)
            
            if sepsis_patients.empty:
                logger.error("未能获取脓毒症患者数据，终止流程")
                return 1
        
        # 步骤2: 获取对照组患者
        control_file = os.path.join(PROCESSED_DIR, "control_patients.csv")
        if os.path.exists(control_file):
            logger.info(f"加载已有的对照组患者数据: {control_file}")
            control_patients = pd.read_csv(control_file)
            if DATA_LIMIT and len(control_patients) > MAX_PATIENTS:
                logger.info(f"限制对照组患者数量为{MAX_PATIENTS}")
                control_patients = control_patients.drop_duplicates(subset=['subject_id']).head(MAX_PATIENTS)
        else:
            logger.info("提取对照组患者数据...")
            # 在获取对照组时限制数量
            control_count = MAX_PATIENTS if DATA_LIMIT else len(sepsis_patients)
            control_patients = get_control_patients(conn, control_count)
        
        # 步骤3: 合并患者数据
        all_patients_file = os.path.join(PROCESSED_DIR, "all_patients_limited.csv") if DATA_LIMIT else os.path.join(PROCESSED_DIR, "all_patients.csv")
        if os.path.exists(all_patients_file):
            logger.info(f"加载已有的合并患者数据: {all_patients_file}")
            all_patients = pd.read_csv(all_patients_file)
        else:
            all_patients = pd.concat([sepsis_patients, control_patients], ignore_index=True)
            # 保留唯一的患者ID
            all_patients = all_patients.drop_duplicates(subset=['subject_id', 'icustay_id'])
            all_patients.to_csv(all_patients_file, index=False)
            logger.info(f"合并后共有 {len(all_patients)} 名患者，数据已保存至 {all_patients_file}")
            
        # 获取唯一的患者ID和入院ID
        unique_stay_ids = all_patients['icustay_id'].unique().tolist()
        unique_hadm_ids = all_patients['hadm_id'].unique().tolist()
        
        logger.info(f"共有 {len(unique_stay_ids)} 个唯一ICU停留和 {len(unique_hadm_ids)} 个唯一入院")
        
        # 现在终止当前正在运行的生命体征数据提取，直接进入时间序列处理阶段
        logger.info("跳过完整的数据提取，直接使用已有数据处理时间序列...")
        
        # 加载已经提取的中间结果
        interim_vitals = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("vital_signs_interim_")])
        if interim_vitals:
            latest_vital_file = os.path.join(PROCESSED_DIR, interim_vitals[-1])
            logger.info(f"使用已有的生命体征中间结果: {latest_vital_file}")
            vital_signs = pd.read_csv(latest_vital_file)
            # 限制每个患者的记录数
            if DATA_LIMIT:
                logger.info(f"限制生命体征数据量...")
                vital_signs = vital_signs.groupby('subject_id').apply(
                    lambda x: x.head(MAX_VITALS_PER_PATIENT)
                ).reset_index(drop=True)
                logger.info(f"限制后的生命体征记录数: {len(vital_signs)}")
        else:
            logger.info("未找到生命体征数据，将使用空DataFrame")
            vital_signs = pd.DataFrame()
        
        # 提取SOFA评分数据 - 但只提取已有患者的数据
        sofa_file = os.path.join(PROCESSED_DIR, "sofa_scores.csv")
        if os.path.exists(sofa_file):
            logger.info(f"加载已有的SOFA评分数据: {sofa_file}")
            sofa_scores = pd.read_csv(sofa_file)
            # 只保留所选患者的数据
            sofa_scores = sofa_scores[sofa_scores['icustay_id'].isin(unique_stay_ids)]
            logger.info(f"筛选后的SOFA评分记录数: {len(sofa_scores)}")
        else:
            logger.info("提取SOFA评分数据，仅限所选患者...")
            sofa_scores = extract_sofa_scores(conn, unique_stay_ids)
        
        # 步骤6: 提取实验室检查数据 - 使用简化方法
        labs_sample_file = os.path.join(PROCESSED_DIR, "lab_values_sample.csv")
        if os.path.exists(labs_sample_file):
            logger.info(f"加载已有的简化实验室检查数据: {labs_sample_file}")
            lab_values = pd.read_csv(labs_sample_file)
        else:
            interim_labs = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("lab_values_interim_")])
            if interim_labs:
                latest_lab_file = os.path.join(PROCESSED_DIR, interim_labs[-1])
                logger.info(f"使用已有的实验室检查中间结果: {latest_lab_file}")
                lab_values = pd.read_csv(latest_lab_file)
            else:
                logger.info("提取少量实验室检查数据样本...")
                # 只提取少量患者的数据
                sample_hadm_ids = unique_hadm_ids[:min(20, len(unique_hadm_ids))]
                lab_values = extract_lab_values(conn, sample_hadm_ids, batch_size=5)
            
            # 限制每个患者的记录数并保存
            if DATA_LIMIT and not lab_values.empty:
                logger.info(f"限制实验室检查数据量...")
                lab_values = lab_values.groupby('subject_id').apply(
                    lambda x: x.head(MAX_LABS_PER_PATIENT)
                ).reset_index(drop=True)
                lab_values.to_csv(labs_sample_file, index=False)
                logger.info(f"限制后的实验室检查记录数: {len(lab_values)}")
        
        # 药物数据 - 简化处理
        meds_sample_file = os.path.join(PROCESSED_DIR, "medications_sample.csv")
        if os.path.exists(meds_sample_file):
            logger.info(f"加载已有的简化药物数据: {meds_sample_file}")
            medication_data = pd.read_csv(meds_sample_file)
        else:
            logger.info("提取少量药物数据样本...")
            # 只提取少量患者的数据
            sample_hadm_ids = unique_hadm_ids[:min(10, len(unique_hadm_ids))]
            medication_data = extract_medication_data(conn, sample_hadm_ids, batch_size=5)
            if not medication_data.empty:
                medication_data.to_csv(meds_sample_file, index=False)
                logger.info(f"保存了 {len(medication_data)} 条药物记录")
        
        # 步骤8: 处理时间序列数据
        ts_sample_file = os.path.join(PROCESSED_DIR, "time_series_sample.csv")
        if os.path.exists(ts_sample_file):
            logger.info(f"加载已有的简化时间序列数据: {ts_sample_file}")
            time_series_data = pd.read_csv(ts_sample_file)
        else:
            logger.info("使用有限数据处理时间序列...")
            time_series_data = process_time_series_data(
                all_patients, vital_signs, lab_values, sofa_scores, medication_data,
                hours_before_sepsis=12, hours_after_sepsis=12  # 减少时间范围
            )
            if not time_series_data.empty:
                time_series_data.to_csv(ts_sample_file, index=False)
                logger.info(f"保存了 {len(time_series_data)} 条时间序列记录")
        
        # 步骤9: 创建训练集、验证集和测试集 - 使用简化数据
        if not time_series_data.empty:
            logger.info("使用简化数据创建训练集、验证集和测试集...")
            train_df, val_df, test_df = create_train_val_test_split(time_series_data, test_size=0.2, val_size=0.2)
            
            # 生成数据统计摘要
            feature_count = len(time_series_data.columns) - 8  # 减去ID和标签列
            logger.info(f"数据集特征数量: {feature_count}")
            
            # 计算缺失值比例
            missing_ratio = time_series_data.isna().mean() * 100
            high_missing_features = missing_ratio[missing_ratio > 50].index.tolist()
            logger.info(f"缺失率超过50%的特征: {', '.join(high_missing_features)}")
        
        # 关闭连接
        conn.close()
        
        # 计算处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"数据提取完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        # 输出提取摘要
        logger.info("============= 数据提取摘要（限制模式）=============")
        logger.info(f"脓毒症患者数: {len(sepsis_patients)}")
        logger.info(f"对照组患者数: {len(control_patients)}")
        logger.info(f"总患者数: {len(all_patients)}")
        logger.info(f"SOFA评分记录数: {len(sofa_scores) if not sofa_scores.empty else 0}")
        logger.info(f"生命体征记录数: {len(vital_signs) if not vital_signs.empty else 0}")
        logger.info(f"实验室检查记录数: {len(lab_values) if not lab_values.empty else 0}")
        logger.info(f"药物记录数: {len(medication_data) if not medication_data.empty else 0}")
        logger.info(f"时间序列记录数: {len(time_series_data) if not time_series_data.empty else 0}")
        logger.info("=======================================")
        
        return 0  # 成功完成
        
    except Exception as e:
        logger.error(f"数据提取过程中出错: {e}")
        traceback.print_exc()
        return 1  # 失败

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIMIC-IV数据提取模块
用于脓毒症早期预警系统

提取并处理各类临床数据：
- 生命体征数据
- 实验室检测结果
- 药物治疗记录
- 临床记录
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import re
from tqdm import tqdm
import sqlalchemy
from sqlalchemy import create_engine
from utils.database_config import get_connection_string, DATABASE_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def connect_to_mimic_db():
    """
    连接到MIMIC-IV数据库
    
    返回:
        psycopg2连接对象
    """
    try:
        import psycopg2
        from utils.database_config import DATABASE_CONFIG, get_connection_string
        
        config = DATABASE_CONFIG
        print(f"尝试连接数据库: {config['host']}:{config['port']}/{config['database']} (用户: {config['user']})")
        
        # 使用连接字符串
        conn_string = get_connection_string()
        print(f"使用连接字符串: {conn_string}")
        
        # 使用最直接的连接方式
        conn = psycopg2.connect(conn_string)
        
        # 测试连接
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"数据库连接成功! PostgreSQL版本: {version}")
        
        # 检查mimiciv schema是否存在
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'mimic%';")
        schemas = cursor.fetchall()
        mimic_schemas = [s[0] for s in schemas]
        print(f"找到MIMIC相关的schema: {mimic_schemas}")
        
        return conn
    except Exception as e:
        print(f"连接数据库失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n尝试诊断连接问题...")
        try:
            # 尝试连接不指定数据库名的简单连接
            import psycopg2
            test_conn = psycopg2.connect(
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port'],
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password']
            )
            print("能够连接到服务器，但可能指定了错误的数据库名")
            test_conn.close()
        except Exception as e2:
            print(f"无法连接到PostgreSQL服务器: {e2}")
            
        raise

def extract_vital_signs(conn, subject_ids, stay_ids, verbose=True):
    """
    从MIMIC-IV中提取生命体征数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        stay_ids: ICU停留ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 生命体征数据
    """
    if verbose:
        logger.info("提取生命体征数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        icu_schema = DATABASE_CONFIG['schema_map']['icu']
    except:
        icu_schema = 'mimiciv_icu'
        logger.warning(f"无法导入数据库配置，使用默认ICU模式: {icu_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    stay_ids_str = ','.join(map(str, stay_ids)) if stay_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND c.subject_id IN ({subject_ids_str})"
    if stay_ids:
        limit_clause += f" AND c.stay_id IN ({stay_ids_str})"
    
    # 查询常见生命体征
    query = f"""
    SELECT 
        c.subject_id, 
        c.stay_id, 
        c.charttime,
        CASE WHEN c.itemid = 220045 THEN 'heart_rate' 
             WHEN c.itemid = 220210 THEN 'respiratory_rate'
             WHEN c.itemid = 220179 THEN 'systolic_bp'
             WHEN c.itemid = 220180 THEN 'diastolic_bp'
             WHEN c.itemid = 220277 THEN 'spo2'
             WHEN c.itemid IN (223761, 223762) THEN 'temperature'
             WHEN c.itemid = 220051 THEN 'mean_bp'
             WHEN c.itemid = 220739 THEN 'gcseye'
             WHEN c.itemid = 223900 THEN 'gcsverbal'
             WHEN c.itemid = 223901 THEN 'gcsmotor'
        END AS vital_sign,
        c.valuenum
    FROM {icu_schema}.chartevents c
    WHERE c.itemid IN 
        (
            220045, -- 心率 
            220210, -- 呼吸频率
            220179, -- 收缩压
            220180, -- 舒张压
            220277, -- 血氧饱和度
            223761, -- 温度（摄氏度）
            223762, -- 温度（华氏度）
            220051, -- 平均动脉压
            220739, -- GCS - 眼睛
            223900, -- GCS - 语言
            223901  -- GCS - 运动
        )
        AND c.valuenum IS NOT NULL
        AND c.valuenum > 0
        {limit_clause}
    ORDER BY c.subject_id, c.stay_id, c.charttime
    """
    
    try:
        if verbose:
            logger.info("执行生命体征查询...")
        vitals_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(vitals_df)} 行生命体征数据")
            logger.info(f"涉及 {vitals_df['subject_id'].nunique()} 名患者")
        
        # 转换华氏温度到摄氏度
        mask = vitals_df['vital_sign'] == 'temperature'
        temp_f_mask = mask & (vitals_df['itemid'] == 223762)
        vitals_df.loc[temp_f_mask, 'valuenum'] = (vitals_df.loc[temp_f_mask, 'valuenum'] - 32) * 5/9
        
        # 计算GCS总分
        gcs_df = vitals_df[vitals_df['vital_sign'].isin(['gcseye', 'gcsverbal', 'gcsmotor'])]
        if not gcs_df.empty:
            gcs_pivot = gcs_df.pivot_table(
                index=['subject_id', 'stay_id', 'charttime'],
                columns='vital_sign',
                values='valuenum',
                aggfunc='mean'
            ).reset_index()
            
            # 填充缺失值并计算总分
            for col in ['gcseye', 'gcsverbal', 'gcsmotor']:
                if col not in gcs_pivot.columns:
                    gcs_pivot[col] = np.nan
            
            gcs_pivot['gcs_total'] = gcs_pivot['gcseye'] + gcs_pivot['gcsverbal'] + gcs_pivot['gcsmotor']
            
            # 转换为与原始数据格式相同的结构
            gcs_total = gcs_pivot[['subject_id', 'stay_id', 'charttime', 'gcs_total']].copy()
            gcs_total['vital_sign'] = 'gcs_total'
            gcs_total.rename(columns={'gcs_total': 'valuenum'}, inplace=True)
            
            # 添加到原始数据
            vitals_df = pd.concat([vitals_df, gcs_total], ignore_index=True)
        
        # 透视表转换为宽格式
        vitals_wide = vitals_df.pivot_table(
            index=['subject_id', 'stay_id', 'charttime'],
            columns='vital_sign',
            values='valuenum',
            aggfunc='mean'
        ).reset_index()
        
        # 转换时间戳到datetime
        vitals_wide['charttime'] = pd.to_datetime(vitals_wide['charttime'])
        
        if verbose:
            logger.info(f"生命体征数据处理完成: {vitals_wide.shape}")
        
        return vitals_wide
        
    except Exception as e:
        logger.error(f"提取生命体征数据时出错: {e}")
        return pd.DataFrame()

def extract_lab_values(conn, subject_ids, hadm_ids, verbose=True):
    """
    从MIMIC-IV中提取实验室检测结果
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        hadm_ids: 住院ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 实验室检测结果
    """
    if verbose:
        logger.info("提取实验室检测数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        hosp_schema = DATABASE_CONFIG['schema_map']['hosp']
    except:
        hosp_schema = 'mimiciv_hosp'
        logger.warning(f"无法导入数据库配置，使用默认医院模式: {hosp_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    hadm_ids_str = ','.join(map(str, hadm_ids)) if hadm_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND le.subject_id IN ({subject_ids_str})"
    if hadm_ids:
        limit_clause += f" AND le.hadm_id IN ({hadm_ids_str})"
    
    # 查询常见脓毒症相关的实验室检测项目
    query = f"""
    SELECT 
        le.subject_id, 
        le.hadm_id, 
        le.charttime,
        CASE 
            WHEN le.itemid = 51301 THEN 'wbc' -- 白细胞计数
            WHEN le.itemid = 50811 THEN 'hemoglobin' -- 血红蛋白
            WHEN le.itemid = 51265 THEN 'platelet' -- 血小板计数
            WHEN le.itemid = 50971 THEN 'bun' -- 血尿素氮
            WHEN le.itemid = 50912 THEN 'creatinine' -- 肌酐
            WHEN le.itemid = 50902 THEN 'chloride' -- 氯离子
            WHEN le.itemid = 50931 THEN 'glucose' -- 葡萄糖
            WHEN le.itemid = 50960 THEN 'bicarbonate' -- 碳酸氢盐
            WHEN le.itemid = 50983 THEN 'sodium' -- 钠
            WHEN le.itemid = 50822 THEN 'potassium' -- 钾
            WHEN le.itemid = 50868 THEN 'anion_gap' -- 阴离子间隙
            WHEN le.itemid = 51144 THEN 'bands' -- 杆状中性粒细胞
            WHEN le.itemid = 51279 THEN 'rdw' -- 红细胞分布宽度
            WHEN le.itemid = 50862 THEN 'albumin' -- 白蛋白
            WHEN le.itemid = 50863 THEN 'alkaline_phosphatase' -- 碱性磷酸酶
            WHEN le.itemid = 50867 THEN 'alt' -- 丙氨酸氨基转移酶
            WHEN le.itemid = 50878 THEN 'ast' -- 天冬氨酸氨基转移酶
            WHEN le.itemid = 51006 THEN 'bilirubin_direct' -- 直接胆红素
            WHEN le.itemid = 50885 THEN 'bilirubin_total' -- 总胆红素
            WHEN le.itemid = 51301 THEN 'wbc' -- 白细胞计数
            WHEN le.itemid = 51221 THEN 'hematocrit' -- 血细胞比容
            WHEN le.itemid = 51237 THEN 'inr' -- 国际标准化比值
            WHEN le.itemid = 51274 THEN 'pt' -- 凝血酶原时间
            WHEN le.itemid = 51275 THEN 'ptt' -- 部分凝血活酶时间
            WHEN le.itemid = 50889 THEN 'calcium' -- 钙
            WHEN le.itemid = 50890 THEN 'calcium_ionized' -- 离子钙
            WHEN le.itemid = 50808 THEN 'chloride' -- 氯
            WHEN le.itemid = 50809 THEN 'potassium' -- 钾
            WHEN le.itemid = 50912 THEN 'creatinine' -- 肌酐
            WHEN le.itemid = 51478 THEN 'troponin_i' -- 肌钙蛋白I
            WHEN le.itemid = 51002 THEN 'troponin_t' -- 肌钙蛋白T
            WHEN le.itemid = 50954 THEN 'lactate' -- 乳酸
            WHEN le.itemid = 50953 THEN 'lactate_dehydrogenase' -- 乳酸脱氢酶
            WHEN le.itemid = 51250 THEN 'neutrophils' -- 中性粒细胞
            WHEN le.itemid = 51254 THEN 'lymphocytes' -- 淋巴细胞 
            WHEN le.itemid = 51244 THEN 'mchc' -- 平均红细胞血红蛋白浓度
            WHEN le.itemid = 51256 THEN 'neutrophils_percent' -- 中性粒细胞百分比
            WHEN le.itemid = 51200 THEN 'base_excess' -- 碱过量
            WHEN le.itemid = 50802 THEN 'base_deficit' -- 碱不足
            WHEN le.itemid = 50804 THEN 'bicarbonate' -- 碳酸氢盐
            WHEN le.itemid = 50813 THEN 'lactate' -- 乳酸
            WHEN le.itemid = 50820 THEN 'ph' -- 酸碱度
            WHEN le.itemid = 50821 THEN 'pco2' -- 二氧化碳分压
            WHEN le.itemid = 50824 THEN 'po2' -- 氧分压
            WHEN le.itemid = 51222 THEN 'pao2_fio2' -- P/F比值
            WHEN le.itemid = 51464 THEN 'procalcitonin' -- 降钙素原
            WHEN le.itemid = 51652 THEN 'crp' -- C反应蛋白
        END AS lab_test,
        le.valuenum
    FROM {hosp_schema}.labevents le
    WHERE le.itemid IN (
        51301, 50811, 51265, 50971, 50912, 50902, 50931, 50960, 50983, 50822, 
        50868, 51144, 51279, 50862, 50863, 50867, 50878, 51006, 50885, 51301, 
        51221, 51237, 51274, 51275, 50889, 50890, 50808, 50809, 50912, 51478, 
        51002, 50954, 50953, 51250, 51254, 51244, 51256, 51200, 50802, 50804, 
        50813, 50820, 50821, 50824, 51222, 51464, 51652
    )
    AND le.valuenum IS NOT NULL
    AND le.valuenum > 0
    {limit_clause}
    ORDER BY le.subject_id, le.hadm_id, le.charttime
    """
    
    try:
        if verbose:
            logger.info("执行实验室检测查询...")
        labs_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(labs_df)} 行实验室检测数据")
            logger.info(f"涉及 {labs_df['subject_id'].nunique()} 名患者")
        
        # 透视表转换为宽格式
        labs_wide = labs_df.pivot_table(
            index=['subject_id', 'hadm_id', 'charttime'],
            columns='lab_test',
            values='valuenum',
            aggfunc='mean'
        ).reset_index()
        
        # 转换时间戳到datetime
        labs_wide['charttime'] = pd.to_datetime(labs_wide['charttime'])
        
        if verbose:
            logger.info(f"实验室检测数据处理完成: {labs_wide.shape}")
            
        return labs_wide
        
    except Exception as e:
        logger.error(f"提取实验室检测数据时出错: {e}")
        return pd.DataFrame()

def extract_antibiotics(conn, subject_ids, hadm_ids, verbose=True):
    """
    从MIMIC-IV中提取抗生素治疗数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        hadm_ids: 住院ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 抗生素治疗数据
    """
    if verbose:
        logger.info("提取抗生素治疗数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        hosp_schema = DATABASE_CONFIG['schema_map']['hosp']
    except:
        hosp_schema = 'mimiciv_hosp'
        logger.warning(f"无法导入数据库配置，使用默认医院模式: {hosp_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    hadm_ids_str = ','.join(map(str, hadm_ids)) if hadm_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND p.subject_id IN ({subject_ids_str})"
    if hadm_ids:
        limit_clause += f" AND p.hadm_id IN ({hadm_ids_str})"
    
    # 查询抗生素数据
    query = f"""
    SELECT 
        p.subject_id, 
        p.hadm_id, 
        p.starttime,
        p.stoptime,
        p.drug AS antibiotic,
        CASE 
            WHEN drug ILIKE '%cefepime%' THEN 'cefepime'
            WHEN drug ILIKE '%ceftriaxone%' THEN 'ceftriaxone'
            WHEN drug ILIKE '%vancomycin%' THEN 'vancomycin'
            WHEN drug ILIKE '%ciprofloxacin%' THEN 'ciprofloxacin'
            WHEN drug ILIKE '%piperacillin%tazobactam%' THEN 'piperacillin_tazobactam'
            WHEN drug ILIKE '%piperacillin%' THEN 'piperacillin'
            WHEN drug ILIKE '%meropenem%' THEN 'meropenem'
            WHEN drug ILIKE '%levofloxacin%' THEN 'levofloxacin'
            WHEN drug ILIKE '%ampicillin%sulbactam%' THEN 'ampicillin_sulbactam'
            WHEN drug ILIKE '%ampicillin%' THEN 'ampicillin'
            WHEN drug ILIKE '%metronidazole%' THEN 'metronidazole'
            WHEN drug ILIKE '%micafungin%' THEN 'micafungin'
            WHEN drug ILIKE '%cefazolin%' THEN 'cefazolin'
            WHEN drug ILIKE '%azithromycin%' THEN 'azithromycin'
            WHEN drug ILIKE '%sulfamethoxazole%trimethoprim%' THEN 'sulfamethoxazole_trimethoprim'
            WHEN drug ILIKE '%daptomycin%' THEN 'daptomycin'
            WHEN drug ILIKE '%linezolid%' THEN 'linezolid'
            WHEN drug ILIKE '%clindamycin%' THEN 'clindamycin'
            WHEN drug ILIKE '%tobramycin%' THEN 'tobramycin'
            WHEN drug ILIKE '%amoxicillin%' THEN 'amoxicillin'
            WHEN drug ILIKE '%amoxicillin%clavulanate%' THEN 'amoxicillin_clavulanate'
            WHEN drug ILIKE '%imipenem%' THEN 'imipenem'
            WHEN drug ILIKE '%imipenem%cilastatin%' THEN 'imipenem_cilastatin'
            WHEN drug ILIKE '%ertapenem%' THEN 'ertapenem'
            WHEN drug ILIKE '%cefpodoxime%' THEN 'cefpodoxime'
            WHEN drug ILIKE '%cephalexin%' THEN 'cephalexin'
            WHEN drug ILIKE '%gentamicin%' THEN 'gentamicin'
            WHEN drug ILIKE '%fluconazole%' THEN 'fluconazole'
            WHEN drug ILIKE '%penicillin%' THEN 'penicillin'
            ELSE 'other'
        END AS antibiotic_class,
        1 AS is_on_antibiotic
    FROM {hosp_schema}.prescriptions p
    WHERE (
        -- 抗生素关键词列表
        p.drug ILIKE '%cefepime%' OR
        p.drug ILIKE '%ceftriaxone%' OR
        p.drug ILIKE '%vancomycin%' OR
        p.drug ILIKE '%ciprofloxacin%' OR
        p.drug ILIKE '%piperacillin%' OR
        p.drug ILIKE '%meropenem%' OR
        p.drug ILIKE '%levofloxacin%' OR
        p.drug ILIKE '%ampicillin%' OR
        p.drug ILIKE '%metronidazole%' OR
        p.drug ILIKE '%micafungin%' OR
        p.drug ILIKE '%cefazolin%' OR
        p.drug ILIKE '%azithromycin%' OR
        p.drug ILIKE '%sulfamethoxazole%trimethoprim%' OR
        p.drug ILIKE '%daptomycin%' OR
        p.drug ILIKE '%linezolid%' OR
        p.drug ILIKE '%clindamycin%' OR
        p.drug ILIKE '%tobramycin%' OR
        p.drug ILIKE '%amoxicillin%' OR
        p.drug ILIKE '%imipenem%' OR
        p.drug ILIKE '%ertapenem%' OR
        p.drug ILIKE '%cefpodoxime%' OR
        p.drug ILIKE '%cephalexin%' OR
        p.drug ILIKE '%gentamicin%' OR
        p.drug ILIKE '%fluconazole%' OR
        p.drug ILIKE '%penicillin%'
    )
    {limit_clause}
    ORDER BY p.subject_id, p.hadm_id, p.starttime
    """
    
    try:
        if verbose:
            logger.info("执行抗生素查询...")
        antibiotics_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(antibiotics_df)} 行抗生素数据")
            logger.info(f"涉及 {antibiotics_df['subject_id'].nunique()} 名患者")
        
        # 转换时间戳到datetime
        antibiotics_df['starttime'] = pd.to_datetime(antibiotics_df['starttime'])
        antibiotics_df['stoptime'] = pd.to_datetime(antibiotics_df['stoptime'])
        
        # 为每个抗生素类创建一个单独的列
        antibiotic_classes = antibiotics_df['antibiotic_class'].unique()
        
        # 创建用于存储时间序列数据的DataFrame
        subject_ids = antibiotics_df['subject_id'].unique()
        hadm_ids = antibiotics_df['hadm_id'].unique()
        
        results = []
        
        for subject_id in subject_ids:
            patient_abx = antibiotics_df[antibiotics_df['subject_id'] == subject_id]
            for hadm_id in patient_abx['hadm_id'].unique():
                hadm_abx = patient_abx[patient_abx['hadm_id'] == hadm_id]
                
                # 确定住院的起止时间范围
                start_time = hadm_abx['starttime'].min()
                end_time = hadm_abx['stoptime'].max()
                
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                
                # 在住院期间按小时创建时间点
                time_range = pd.date_range(start=start_time, end=end_time, freq='1H')
                
                for time_point in time_range:
                    row = {'subject_id': subject_id, 'hadm_id': hadm_id, 'charttime': time_point}
                    
                    # 检查该时间点使用的抗生素
                    for abx_class in antibiotic_classes:
                        abx_at_time = hadm_abx[
                            (hadm_abx['antibiotic_class'] == abx_class) &
                            (hadm_abx['starttime'] <= time_point) &
                            (hadm_abx['stoptime'] >= time_point)
                        ]
                        row[f'abx_{abx_class}'] = 1 if not abx_at_time.empty else 0
                    
                    results.append(row)
        
        antibiotics_time_df = pd.DataFrame(results)
        
        if verbose and not antibiotics_time_df.empty:
            logger.info(f"抗生素数据处理完成: {antibiotics_time_df.shape}")
            
        return antibiotics_time_df
        
    except Exception as e:
        logger.error(f"提取抗生素数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def extract_vasopressors(conn, subject_ids, stay_ids, verbose=True):
    """
    从MIMIC-IV中提取升压药使用数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        stay_ids: ICU停留ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 升压药使用数据
    """
    if verbose:
        logger.info("提取升压药使用数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        icu_schema = DATABASE_CONFIG['schema_map']['icu']
    except:
        icu_schema = 'mimiciv_icu'
        logger.warning(f"无法导入数据库配置，使用默认ICU模式: {icu_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    stay_ids_str = ','.join(map(str, stay_ids)) if stay_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND i.subject_id IN ({subject_ids_str})"
    if stay_ids:
        limit_clause += f" AND i.stay_id IN ({stay_ids_str})"
    
    # 查询升压药数据
    query = f"""
    SELECT 
        i.subject_id, 
        i.stay_id, 
        i.starttime,
        i.endtime,
        i.itemid,
        CASE 
            WHEN i.itemid = 221906 THEN 'norepinephrine'
            WHEN i.itemid = 221289 THEN 'epinephrine'
            WHEN i.itemid = 222315 THEN 'vasopressin'
            WHEN i.itemid = 221749 THEN 'phenylephrine'
            WHEN i.itemid = 221662 THEN 'dopamine'
            WHEN i.itemid = 221653 THEN 'dobutamine'
        END AS vasopressor,
        i.amount,
        i.amountuom,
        i.rate,
        i.rateuom
    FROM {icu_schema}.inputevents i
    WHERE i.itemid IN (
        221906, -- norepinephrine
        221289, -- epinephrine
        222315, -- vasopressin
        221749, -- phenylephrine
        221662, -- dopamine
        221653  -- dobutamine
    )
    AND i.amount IS NOT NULL
    AND i.rate IS NOT NULL
    AND i.rate > 0
    {limit_clause}
    ORDER BY i.subject_id, i.stay_id, i.starttime
    """
    
    try:
        if verbose:
            logger.info("执行升压药查询...")
        vaso_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(vaso_df)} 行升压药数据")
            logger.info(f"涉及 {vaso_df['subject_id'].nunique()} 名患者")
        
        if vaso_df.empty:
            return pd.DataFrame()
        
        # 转换时间戳到datetime
        vaso_df['starttime'] = pd.to_datetime(vaso_df['starttime'])
        vaso_df['endtime'] = pd.to_datetime(vaso_df['endtime'])
        
        # 创建用于存储时间序列数据的DataFrame
        results = []
        
        for subject_id in vaso_df['subject_id'].unique():
            patient_vaso = vaso_df[vaso_df['subject_id'] == subject_id]
            for stay_id in patient_vaso['stay_id'].unique():
                stay_vaso = patient_vaso[patient_vaso['stay_id'] == stay_id]
                
                # 确定ICU停留的起止时间范围
                start_time = stay_vaso['starttime'].min()
                end_time = stay_vaso['endtime'].max()
                
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                
                # 在ICU停留期间按小时创建时间点
                time_range = pd.date_range(start=start_time, end=end_time, freq='1H')
                
                for time_point in time_range:
                    row = {'subject_id': subject_id, 'stay_id': stay_id, 'charttime': time_point}
                    
                    # 对每种升压药，检查在该时间点的使用情况
                    for vaso_type in vaso_df['vasopressor'].unique():
                        vaso_at_time = stay_vaso[
                            (stay_vaso['vasopressor'] == vaso_type) &
                            (stay_vaso['starttime'] <= time_point) &
                            (stay_vaso['endtime'] >= time_point)
                        ]
                        
                        # 如果有使用该升压药，计算剂量
                        if not vaso_at_time.empty:
                            # 取最大剂量
                            max_rate = vaso_at_time['rate'].max()
                            row[f'vaso_{vaso_type}'] = max_rate
                        else:
                            row[f'vaso_{vaso_type}'] = 0
                    
                    results.append(row)
        
        vasopressors_time_df = pd.DataFrame(results)
        
        if verbose and not vasopressors_time_df.empty:
            logger.info(f"升压药数据处理完成: {vasopressors_time_df.shape}")
        
        return vasopressors_time_df
        
    except Exception as e:
        logger.error(f"提取升压药数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def extract_diagnoses(conn, subject_ids, hadm_ids, verbose=True):
    """
    从MIMIC-IV中提取诊断数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        hadm_ids: 住院ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 诊断数据
    """
    if verbose:
        logger.info("提取诊断数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        hosp_schema = DATABASE_CONFIG['schema_map']['hosp']
    except:
        hosp_schema = 'mimiciv_hosp'
        logger.warning(f"无法导入数据库配置，使用默认医院模式: {hosp_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    hadm_ids_str = ','.join(map(str, hadm_ids)) if hadm_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND d.subject_id IN ({subject_ids_str})"
    if hadm_ids:
        limit_clause += f" AND d.hadm_id IN ({hadm_ids_str})"
    
    # 查询ICD诊断
    query = f"""
    SELECT 
        d.subject_id, 
        d.hadm_id, 
        d.icd_code,
        d.icd_version,
        i.long_title as diagnosis_description,
        CASE
            WHEN d.icd_code LIKE 'A%' OR d.icd_code LIKE 'B%' THEN 'infectious'
            WHEN d.icd_code LIKE 'I%' THEN 'cardiovascular'
            WHEN d.icd_code LIKE 'J%' THEN 'respiratory'
            WHEN d.icd_code LIKE 'K%' THEN 'digestive'
            WHEN d.icd_code LIKE 'N%' THEN 'genitourinary'
            WHEN d.icd_code LIKE 'C%' OR d.icd_code LIKE 'D0%' OR d.icd_code LIKE 'D1%' OR d.icd_code LIKE 'D2%' OR d.icd_code LIKE 'D3%' OR d.icd_code LIKE 'D4%' THEN 'neoplasms'
            WHEN d.icd_code LIKE 'E%' THEN 'endocrine'
            WHEN d.icd_code LIKE 'F%' THEN 'mental'
            WHEN d.icd_code LIKE 'G%' THEN 'nervous'
            WHEN d.icd_code LIKE 'H0%' OR d.icd_code LIKE 'H1%' OR d.icd_code LIKE 'H2%' OR d.icd_code LIKE 'H3%' OR d.icd_code LIKE 'H4%' OR d.icd_code LIKE 'H5%' THEN 'eye'
            WHEN d.icd_code LIKE 'H6%' OR d.icd_code LIKE 'H7%' OR d.icd_code LIKE 'H8%' OR d.icd_code LIKE 'H9%' THEN 'ear'
            WHEN d.icd_code LIKE 'L%' THEN 'skin'
            WHEN d.icd_code LIKE 'M%' THEN 'musculoskeletal'
            WHEN d.icd_code LIKE 'O%' THEN 'pregnancy'
            WHEN d.icd_code LIKE 'P%' THEN 'perinatal'
            WHEN d.icd_code LIKE 'Q%' THEN 'congenital'
            WHEN d.icd_code LIKE 'R%' THEN 'abnormal_findings'
            WHEN d.icd_code LIKE 'S%' OR d.icd_code LIKE 'T%' THEN 'injury_poisoning'
            WHEN d.icd_code LIKE 'V%' OR d.icd_code LIKE 'W%' OR d.icd_code LIKE 'X%' OR d.icd_code LIKE 'Y%' THEN 'external_causes'
            WHEN d.icd_code LIKE 'Z%' THEN 'health_status'
            ELSE 'other'
        END AS diagnosis_category,
        CASE 
            WHEN d.icd_code LIKE 'A40%' OR d.icd_code LIKE 'A41%' THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code = '99591' OR d.icd_code = '99592') THEN 1
            WHEN d.icd_version = 10 AND (d.icd_code = 'R6520' OR d.icd_code = 'R6521') THEN 1
            ELSE 0
        END AS is_sepsis,
        CASE 
            WHEN (d.icd_code LIKE 'J1%' OR d.icd_code LIKE 'J2%') THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code LIKE '48%' OR d.icd_code LIKE '481%') THEN 1
            ELSE 0
        END AS is_pneumonia,
        CASE 
            WHEN (d.icd_code LIKE 'N39.0%') THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code = '5990') THEN 1
            ELSE 0
        END AS is_uti,
        CASE 
            WHEN d.icd_code LIKE 'K65%' THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code LIKE '567%') THEN 1
            ELSE 0
        END AS is_peritonitis,
        CASE 
            WHEN d.icd_code LIKE 'A40%' OR d.icd_code LIKE 'A41%' THEN 1
            WHEN d.icd_code LIKE 'J1%' OR d.icd_code LIKE 'J2%' THEN 1
            WHEN d.icd_code LIKE 'N39.0%' THEN 1
            WHEN d.icd_code LIKE 'K65%' THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code = '99591' OR d.icd_code = '99592') THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code LIKE '48%' OR d.icd_code LIKE '481%') THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code = '5990') THEN 1
            WHEN d.icd_version = 9 AND (d.icd_code LIKE '567%') THEN 1
            WHEN d.icd_version = 10 AND (d.icd_code = 'R6520' OR d.icd_code = 'R6521') THEN 1
            ELSE 0
        END AS is_infection
    FROM {hosp_schema}.diagnoses_icd d
    LEFT JOIN {hosp_schema}.d_icd_diagnoses i
    ON d.icd_code = i.icd_code
    AND d.icd_version = i.icd_version
    WHERE d.hadm_id IS NOT NULL
    {limit_clause}
    ORDER BY d.subject_id, d.hadm_id
    """
    
    try:
        if verbose:
            logger.info("执行诊断查询...")
        diagnoses_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(diagnoses_df)} 行诊断数据")
            logger.info(f"涉及 {diagnoses_df['subject_id'].nunique()} 名患者")
        
        # 按患者和住院ID分组，汇总诊断类别
        diagnoses_summary = diagnoses_df.groupby(['subject_id', 'hadm_id']).agg({
            'is_sepsis': 'max',
            'is_pneumonia': 'max',
            'is_uti': 'max',
            'is_peritonitis': 'max',
            'is_infection': 'max'
        }).reset_index()
        
        # 统计每个住院的诊断类别分布
        category_counts = diagnoses_df.groupby(['subject_id', 'hadm_id', 'diagnosis_category']).size().unstack(fill_value=0).reset_index()
        
        # 合并诊断汇总和类别分布
        diagnoses_features = pd.merge(diagnoses_summary, category_counts, on=['subject_id', 'hadm_id'], how='inner')
        
        if verbose:
            logger.info(f"诊断数据处理完成: {diagnoses_features.shape}")
            
        return diagnoses_features
        
    except Exception as e:
        logger.error(f"提取诊断数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def extract_nursing_notes(conn, subject_ids, stay_ids, verbose=True):
    """
    从MIMIC-IV中提取护理记录数据
    
    参数:
        conn: 数据库连接
        subject_ids: 患者ID列表
        stay_ids: ICU停留ID列表
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 护理记录数据
    """
    if verbose:
        logger.info("提取护理记录数据...")
    
    # 获取数据库模式
    try:
        from utils.database_config import DATABASE_CONFIG
        icu_schema = DATABASE_CONFIG['schema_map']['icu']
    except:
        icu_schema = 'mimiciv_icu'
        logger.warning(f"无法导入数据库配置，使用默认ICU模式: {icu_schema}")
    
    # 转换ID列表为字符串格式，用于SQL查询
    subject_ids_str = ','.join(map(str, subject_ids)) if subject_ids else 'NULL'
    stay_ids_str = ','.join(map(str, stay_ids)) if stay_ids else 'NULL'
    
    # 限制条件
    limit_clause = ""
    if subject_ids:
        limit_clause += f" AND n.subject_id IN ({subject_ids_str})"
    if stay_ids:
        limit_clause += f" AND n.stay_id IN ({stay_ids_str})"
    
    # 查询护理记录
    query = f"""
    SELECT 
        n.subject_id, 
        n.stay_id, 
        n.charttime,
        n.text,
        CASE 
            WHEN n.category = 'Nursing/other' THEN 'nursing_note'
            WHEN n.category = 'Nursing Progress Note' THEN 'progress_note'
            WHEN n.category = 'Physician ' THEN 'physician_note'
            ELSE LOWER(n.category)
        END AS note_type
    FROM {icu_schema}.noteevents n
    WHERE n.category IN (
        'Nursing/other',
        'Nursing Progress Note',
        'Physician ',
        'Respiratory ',
        'General',
        'Nutrition',
        'Echo',
        'ECG',
        'Consult'
    )
    AND n.text IS NOT NULL
    AND LENGTH(n.text) > 10
    {limit_clause}
    ORDER BY n.subject_id, n.stay_id, n.charttime
    """
    
    try:
        if verbose:
            logger.info("执行护理记录查询...")
        notes_df = pd.read_sql(query, conn)
        
        if verbose:
            logger.info(f"提取了 {len(notes_df)} 行护理记录数据")
            logger.info(f"涉及 {notes_df['subject_id'].nunique()} 名患者")
        
        # 转换时间戳到datetime
        notes_df['charttime'] = pd.to_datetime(notes_df['charttime'])
        
        # 清理和预处理护理记录文本
        notes_df['text'] = notes_df['text'].str.replace(r'\[\*\*.*?\*\*\]', '', regex=True)  # 移除脱敏标记
        notes_df['text'] = notes_df['text'].str.replace(r'\n', ' ', regex=True)  # 替换换行符
        notes_df['text'] = notes_df['text'].str.replace(r'\s+', ' ', regex=True)  # 删除多余空格
        notes_df['text'] = notes_df['text'].str.strip()  # 删除首尾空格
        
        if verbose:
            logger.info(f"护理记录数据处理完成: {notes_df.shape}")
            
        return notes_df
        
    except Exception as e:
        logger.error(f"提取护理记录数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def align_time_series(vitals_df, labs_df, antibiotics_df, vasopressors_df, hourly_bins=True, verbose=True):
    """
    对齐不同来源的时间序列数据
    
    参数:
        vitals_df: 生命体征数据
        labs_df: 实验室检测数据
        antibiotics_df: 抗生素使用数据
        vasopressors_df: 升压药使用数据
        hourly_bins: 是否按小时对齐
        verbose: 是否显示进度信息
        
    返回:
        DataFrame: 对齐后的时间序列数据
    """
    if verbose:
        logger.info("开始对齐时间序列数据...")
    
    # 检查输入数据
    if vitals_df.empty and labs_df.empty and antibiotics_df.empty and vasopressors_df.empty:
        logger.warning("所有输入数据都为空，无法对齐时间序列")
        return pd.DataFrame()
    
    # 收集所有数据源中的患者ID
    subject_ids = set()
    for df in [vitals_df, labs_df, antibiotics_df, vasopressors_df]:
        if not df.empty:
            subject_ids.update(df['subject_id'].unique())
    
    if verbose:
        logger.info(f"共有 {len(subject_ids)} 名患者的数据需要对齐")
    
    all_aligned_data = []
    
    # 为每个患者对齐数据
    for subject_id in tqdm(subject_ids, desc="对齐患者数据", disable=not verbose):
        # 收集该患者在不同数据源中的ID
        patient_stay_ids = set()
        patient_hadm_ids = set()
        
        # 收集ICU停留ID
        for df in [vitals_df, vasopressors_df]:
            if not df.empty and 'stay_id' in df.columns:
                patient_df = df[df['subject_id'] == subject_id]
                if not patient_df.empty:
                    patient_stay_ids.update(patient_df['stay_id'].unique())
        
        # 收集住院ID
        for df in [labs_df, antibiotics_df]:
            if not df.empty and 'hadm_id' in df.columns:
                patient_df = df[df['subject_id'] == subject_id]
                if not patient_df.empty:
                    patient_hadm_ids.update(patient_df['hadm_id'].unique())
        
        # 如果没有找到任何ID，跳过该患者
        if not patient_stay_ids and not patient_hadm_ids:
            continue
        
        # 为该患者的每个ICU停留处理数据
        for stay_id in patient_stay_ids:
            # 筛选该ICU停留的数据
            patient_vitals = vitals_df[(vitals_df['subject_id'] == subject_id) & (vitals_df['stay_id'] == stay_id)] if not vitals_df.empty else pd.DataFrame()
            patient_vaso = vasopressors_df[(vasopressors_df['subject_id'] == subject_id) & (vasopressors_df['stay_id'] == stay_id)] if not vasopressors_df.empty else pd.DataFrame()
            
            # 找到对应的住院ID
            related_hadm_ids = set()
            for df in [labs_df, antibiotics_df]:
                if not df.empty and 'hadm_id' in df.columns and 'stay_id' in df.columns:
                    related = df[(df['subject_id'] == subject_id) & (df['stay_id'] == stay_id)]
                    if not related.empty:
                        related_hadm_ids.update(related['hadm_id'].unique())
            
            # 如果没有找到对应的住院ID，尝试通过时间匹配
            if not related_hadm_ids and not patient_vitals.empty and (not labs_df.empty or not antibiotics_df.empty):
                icu_start_time = patient_vitals['charttime'].min()
                icu_end_time = patient_vitals['charttime'].max()
                
                for df in [labs_df, antibiotics_df]:
                    if not df.empty and 'hadm_id' in df.columns and 'charttime' in df.columns:
                        patient_df = df[df['subject_id'] == subject_id]
                        if not patient_df.empty:
                            # 检查有数据点落在ICU停留时间范围内的住院
                            overlapping = patient_df[
                                (patient_df['charttime'] >= icu_start_time) &
                                (patient_df['charttime'] <= icu_end_time)
                            ]
                            if not overlapping.empty:
                                related_hadm_ids.update(overlapping['hadm_id'].unique())
            
            # 为每个相关的住院ID处理数据
            for hadm_id in related_hadm_ids:
                patient_labs = labs_df[(labs_df['subject_id'] == subject_id) & (labs_df['hadm_id'] == hadm_id)] if not labs_df.empty else pd.DataFrame()
                patient_abx = antibiotics_df[(antibiotics_df['subject_id'] == subject_id) & (antibiotics_df['hadm_id'] == hadm_id)] if not antibiotics_df.empty else pd.DataFrame()
                
                # 确定时间范围
                all_times = []
                for df in [patient_vitals, patient_labs, patient_abx, patient_vaso]:
                    if not df.empty and 'charttime' in df.columns:
                        all_times.extend(df['charttime'].tolist())
                
                if not all_times:
                    continue
                
                min_time = min(all_times)
                max_time = max(all_times)
                
                # 创建时间索引
                if hourly_bins:
                    time_range = pd.date_range(start=min_time.floor('H'), end=max_time.ceil('H'), freq='1H')
                else:
                    # 使用所有不同的时间点
                    time_range = sorted(set(all_times))
                
                # 初始化结果DataFrame
                aligned_data = pd.DataFrame({
                    'subject_id': subject_id,
                    'stay_id': stay_id,
                    'hadm_id': hadm_id,
                    'charttime': time_range
                })
                
                # 合并数据
                # 生命体征
                if not patient_vitals.empty:
                    # 对于每个时间点，找到最接近的生命体征记录
                    for time_point in time_range:
                        closest_idx = (patient_vitals['charttime'] - time_point).abs().idxmin() if not patient_vitals.empty else None
                        if closest_idx is not None:
                            time_diff = abs((patient_vitals.loc[closest_idx, 'charttime'] - time_point).total_seconds()) / 3600
                            # 只使用2小时内的最接近记录
                            if time_diff <= 2:
                                for col in patient_vitals.columns:
                                    if col not in ['subject_id', 'stay_id', 'hadm_id', 'charttime']:
                                        aligned_data.loc[aligned_data['charttime'] == time_point, col] = patient_vitals.loc[closest_idx, col]
                
                # 实验室检测
                if not patient_labs.empty:
                    # 对于每个时间点，找到最接近的实验室检测记录
                    for time_point in time_range:
                        closest_idx = (patient_labs['charttime'] - time_point).abs().idxmin() if not patient_labs.empty else None
                        if closest_idx is not None:
                            time_diff = abs((patient_labs.loc[closest_idx, 'charttime'] - time_point).total_seconds()) / 3600
                            # 只使用24小时内的最接近记录（实验室检测可能更不频繁）
                            if time_diff <= 24:
                                for col in patient_labs.columns:
                                    if col not in ['subject_id', 'stay_id', 'hadm_id', 'charttime'] and col not in aligned_data.columns:
                                        aligned_data.loc[aligned_data['charttime'] == time_point, col] = patient_labs.loc[closest_idx, col]
                
                # 抗生素使用
                if not patient_abx.empty:
                    for col in patient_abx.columns:
                        if col.startswith('abx_') and col not in aligned_data.columns:
                            aligned_data[col] = 0
                    
                    for idx, row in patient_abx.iterrows():
                        time_point = row['charttime']
                        for col in patient_abx.columns:
                            if col.startswith('abx_') and row[col] == 1:
                                closest_idx = (aligned_data['charttime'] - time_point).abs().idxmin()
                                if closest_idx is not None:
                                    aligned_data.loc[closest_idx, col] = 1
                
                # 升压药使用
                if not patient_vaso.empty:
                    for col in patient_vaso.columns:
                        if col.startswith('vaso_') and col not in aligned_data.columns:
                            aligned_data[col] = 0
                    
                    for idx, row in patient_vaso.iterrows():
                        time_point = row['charttime']
                        for col in patient_vaso.columns:
                            if col.startswith('vaso_'):
                                closest_idx = (aligned_data['charttime'] - time_point).abs().idxmin()
                                if closest_idx is not None:
                                    aligned_data.loc[closest_idx, col] = row[col]
                
                # 添加到结果
                all_aligned_data.append(aligned_data)
    
    # 合并所有患者的对齐数据
    if all_aligned_data:
        final_aligned_data = pd.concat(all_aligned_data, ignore_index=True)
        
        # 填充缺失值
        numeric_cols = final_aligned_data.select_dtypes(include=['number']).columns
        # 向前填充数值
        final_aligned_data[numeric_cols] = final_aligned_data.groupby(['subject_id', 'stay_id', 'hadm_id'])[numeric_cols].fillna(method='ffill')
        
        if verbose:
            logger.info(f"时间序列数据对齐完成: {final_aligned_data.shape}")
            logger.info(f"包含 {final_aligned_data['subject_id'].nunique()} 名患者, {final_aligned_data['stay_id'].nunique()} 个ICU停留")
        
        return final_aligned_data
    else:
        logger.warning("没有可对齐的数据")
        return pd.DataFrame()
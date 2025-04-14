# 脓毒症早期预警系统 - 数据提取主脚本
# -*- coding: utf-8 -*-

# 在脚本开头添加以下内容
import os
import sys
import time
import psycopg2
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import traceback
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 现在尝试导入
try:
    from utils.database_config import get_connection_string, DATABASE_CONFIG, get_query_optimizations
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error(f"当前Python路径: {sys.path}")
    
    # 尝试直接从utils导入
    utils_dir = os.path.join(project_root, 'utils')
    sys.path.append(utils_dir)
    try:
        from database_config import get_connection_string, DATABASE_CONFIG, get_query_optimizations
        logger.info("从utils目录直接导入成功")
    except ImportError:
        logger.error("无法导入database_config，使用默认配置")
        
        # 创建默认配置，让脚本可以继续运行
        DATABASE_CONFIG = {
            'host': '172.16.3.67',
            'port': 5432,
            'database': 'mimiciv',
            'user': 'postgres',
            'password': '123456',
            'schema_map': {
                'hosp': 'mimiciv_hosp',
                'icu': 'mimiciv_icu',
                'derived': 'mimiciv_derived'
            },
            'min_patient_count': 1000,
            'use_mock_data': True
        }
        
        def get_connection_string():
            return ""
            
        def get_query_optimizations():
            return {}

try:
    from utils.data_extraction import extract_vital_signs, extract_lab_values
    from utils.data_extraction import extract_antibiotics, extract_vasopressors, extract_diagnoses
    from utils.data_extraction import extract_nursing_notes, align_time_series
    from models.text.text_processing import process_clinical_text, encode_clinical_text
    from models.knowledge_graph.knowledge_graph import build_medical_kg, convert_nx_to_pytorch_geometric, generate_kg_embeddings
    from utils.data_loading import generate_sample_data
except ImportError as e:
    logger.error(f"导入功能模块失败: {e}")

def optimize_db_connection(conn):
    """
    Optimize database connection for faster queries
    
    Args:
        conn: Database connection
    """
    try:
        from utils.database_config import get_query_optimizations
        optimizations = get_query_optimizations()
        
        cursor = conn.cursor()
        
        # Apply optimizations
        logger.info("Applying database optimization settings...")
        for param, value in optimizations.items():
            # Ensure work_mem is within PostgreSQL limits (64KB - 2097151KB)
            if param == 'work_mem' and ('GB' in value or 'gb' in value):
                # Convert GB to MB to stay within limits
                if 'GB' in value:
                    num_gb = float(value.replace('GB', '').strip())
                    if num_gb >= 2:
                        value = '1999MB'  # Just under 2GB limit
                    else:
                        value = f"{int(num_gb * 1000)}MB"
                        
            # Apply setting        
            try:
                cursor.execute(f"SET {param} = '{value}'")
                logger.debug(f"Set {param} = {value}")
            except Exception as e:
                logger.warning(f"Failed to set {param} = {value}: {e}")
        
        cursor.close()
        logger.info("Database optimization settings applied")
    except Exception as e:
        logger.warning(f"Failed to optimize database connection: {e}")
        # Continue execution even if optimization fails

def get_all_icu_patients(conn, min_stay_hours=24):
    """
    获取所有符合条件的ICU患者
    
    参数:
        conn: 数据库连接
        min_stay_hours: 最小ICU停留时间（小时）
        
    返回:
        DataFrame，包含患者ID和入院ID
    """
    icu_schema = DATABASE_CONFIG['schema_map']['icu']
    
    query = f"""
    SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id,
           i.intime, i.outtime,
           EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 AS stay_hours
    FROM {icu_schema}.icustays i
    WHERE EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600 >= {min_stay_hours}
    ORDER BY i.subject_id, i.hadm_id, i.stay_id
    """
    
    return pd.read_sql(query, conn)

def get_sepsis_patients(conn, logger):
    """
    从MIMIC-IV数据库中检索脓毒症患者数据。
    
    Args:
        conn: 数据库连接对象
        logger: 日志记录器对象
    
    Returns:
        包含脓毒症患者信息的DataFrame
    """
    try:
        logger.info("获取脓毒症患者数据...")
        cursor = conn.cursor()
        
        # 查询脓毒症患者数据
        query = """
        SELECT  s.subject_id,
                s.hadm_id,
                s.icustay_id as stay_id,
                s.infection_time as sepsis_onset_time,
                s.sepsis3
        FROM    mimiciv_derived.sepsis3 s
        WHERE   s.sepsis3 = 1
                AND s.sepsis3_onset_order = 1
        ORDER BY s.icustay_id
        """
        
        cursor.execute(query)
        result = cursor.fetchall()
        
        # 创建结果DataFrame
        df = pd.DataFrame(result, columns=['subject_id', 'hadm_id', 'stay_id', 'sepsis_onset_time', 'sepsis3'])
        logger.info(f"检索到 {len(df)} 名脓毒症患者")
        return df
    
    except Exception as e:
        logger.error(f"获取脓毒症患者数据时出错: {e}")
        raise

def process_patient_chunk(patient_chunk, connection_string):
    """
    处理一批患者数据
    
    参数:
        patient_chunk: 包含患者ID的DataFrame
        connection_string: 数据库连接字符串
        
    返回:
        处理后的患者数据
    """
    try:
        # 建立数据库连接
        conn = psycopg2.connect(connection_string)
        
        # 优化连接
        optimize_db_connection(conn)
        
        # 提取患者信息
        subject_ids = patient_chunk['subject_id'].tolist()
        hadm_ids = patient_chunk['hadm_id'].tolist()
        stay_ids = patient_chunk['stay_id'].tolist()
        
        # 提取各类数据
        vitals = extract_vital_signs(conn, subject_ids, stay_ids)
        labs = extract_lab_values(conn, subject_ids, hadm_ids)
        antibiotics = extract_antibiotics(conn, subject_ids, hadm_ids)
        vasopressors = extract_vasopressors(conn, subject_ids, stay_ids)
        diagnoses = extract_diagnoses(conn, subject_ids, hadm_ids)
        notes = extract_nursing_notes(conn, subject_ids, stay_ids)
        
        # 对齐时间序列数据
        aligned_data = align_time_series(vitals, labs, antibiotics, vasopressors)
        
        # 关闭连接
        conn.close()
        
        return {
            'patient_info': patient_chunk,
            'aligned_data': aligned_data,
            'diagnoses': diagnoses,
            'notes': notes
        }
    except Exception as e:
        print(f"处理患者批次时出错: {e}")
        traceback.print_exc()
        return None

def main():
    """
    主函数：提取MIMIC-IV数据用于脓毒症预测
    """
    start_time = time.time()
    print(f"开始数据提取流程，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 连接到数据库
        conn_string = get_connection_string()
        conn = psycopg2.connect(conn_string)
        
        # 优化连接
        optimize_db_connection(conn)
        
        # 获取所有ICU患者
        print("获取符合条件的ICU患者...")
        icu_patients = get_all_icu_patients(conn, min_stay_hours=24)
        print(f"找到 {len(icu_patients)} 名符合条件的ICU患者")
        
        # 获取脓毒症患者
        print("获取脓毒症患者...")
        sepsis_patients = get_sepsis_patients(conn, logger)
        print(f"找到 {len(sepsis_patients)} 名脓毒症患者")
        
        # 将脓毒症发作时间合并到ICU患者表中
        sepsis_info = sepsis_patients[['subject_id', 'hadm_id', 'stay_id', 'sepsis_onset_time']]
        icu_patients = pd.merge(
            icu_patients, 
            sepsis_info, 
            on=['subject_id', 'hadm_id', 'stay_id'], 
            how='left'
        )
        
        # 标记脓毒症患者
        icu_patients['sepsis_label'] = icu_patients['sepsis_onset_time'].notnull().astype(int)
        
        # 确保数据量足够大
        min_patient_count = DATABASE_CONFIG.get('min_patient_count', 5000)
        if len(icu_patients) < min_patient_count:
            print(f"警告：找到的患者数量 ({len(icu_patients)}) 少于目标数量 ({min_patient_count})")
            print("尝试放宽筛选条件...")
            
            # 降低ICU停留时间要求，获取更多患者
            icu_patients = get_all_icu_patients(conn, min_stay_hours=12)
            
            # 重新合并脓毒症信息
            icu_patients = pd.merge(
                icu_patients, 
                sepsis_info, 
                on=['subject_id', 'hadm_id', 'stay_id'], 
                how='left'
            )
            icu_patients['sepsis_label'] = icu_patients['sepsis_onset_time'].notnull().astype(int)
            
            print(f"调整后找到 {len(icu_patients)} 名患者")
        
        # 关闭连接
        conn.close()
        
        # 准备并行处理患者数据
        print("开始并行提取患者数据...")
        
        # 确定并行处理的核心数
        num_cores = min(cpu_count(), 16)  # 最多使用16个核心
        print(f"使用 {num_cores} 个CPU核心并行处理")
        
        # 将患者分块处理
        chunk_size = min(5000, max(1000, len(icu_patients) // num_cores))
        patient_chunks = [icu_patients.iloc[i:i+chunk_size] for i in range(0, len(icu_patients), chunk_size)]
        print(f"将患者分为 {len(patient_chunks)} 个批次处理")
        
        # 并行处理
        with Pool(num_cores) as pool:
            results = pool.starmap(
                process_patient_chunk,
                [(chunk, conn_string) for chunk in patient_chunks]
            )
        
        # 合并结果
        print("合并处理结果...")
        all_patient_info = []
        all_aligned_data = []
        all_diagnoses = []
        all_notes = []
        
        for result in results:
            if result:
                all_patient_info.append(result['patient_info'])
                all_aligned_data.append(result['aligned_data'])
                all_diagnoses.append(result['diagnoses'])
                all_notes.append(result['notes'])
        
        # 连接所有数据
        patient_info = pd.concat(all_patient_info, ignore_index=True)
        aligned_data = pd.concat(all_aligned_data, ignore_index=True)
        diagnoses = pd.concat(all_diagnoses, ignore_index=True) if all_diagnoses else pd.DataFrame()
        notes = pd.concat(all_notes, ignore_index=True) if all_notes else pd.DataFrame()
        
        # 创建输出目录
        os.makedirs('data/processed', exist_ok=True)
        
        # 保存结果
        print("保存处理后的数据...")
        patient_info.to_csv('data/processed/patient_info.csv', index=False)
        aligned_data.to_csv('data/processed/aligned_data.csv', index=False)
        
        if not diagnoses.empty:
            diagnoses.to_csv('data/processed/diagnoses.csv', index=False)
        
        if not notes.empty:
            notes.to_csv('data/processed/nursing_notes.csv', index=False)
        
        # 分别保存脓毒症和非脓毒症患者的数据
        # 便于后续分析和模型训练
        sepsis_patients = aligned_data[aligned_data['subject_id'].isin(
            patient_info[patient_info['sepsis_label'] == 1]['subject_id']
        )]
        non_sepsis_patients = aligned_data[aligned_data['subject_id'].isin(
            patient_info[patient_info['sepsis_label'] == 0]['subject_id']
        )]
        
        sepsis_patients.to_csv('data/processed/sepsis_patient_data.csv', index=False)
        non_sepsis_patients.to_csv('data/processed/non_sepsis_patient_data.csv', index=False)
        
        # 输出统计信息
        end_time = time.time()
        duration = (end_time - start_time) / 60  # 转换为分钟
        
        print("\n数据提取完成!")
        print(f"总耗时: {duration:.2f}分钟")
        print(f"处理的ICU患者总数: {len(patient_info)}")
        print(f"脓毒症患者数: {len(patient_info[patient_info['sepsis_label'] == 1])}")
        print(f"非脓毒症患者数: {len(patient_info[patient_info['sepsis_label'] == 0])}")
        print(f"数据点总数: {len(aligned_data)}")
        print(f"特征总数: {len(aligned_data.columns) - 3}")  # 减去subject_id, hadm_id, charttime
        
        print("\n数据已保存到 data/processed/ 目录")
        
    except Exception as e:
        print(f"数据提取过程出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
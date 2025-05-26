#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块，处理MIMIC数据集中的多模态数据
简化版用于加载插补后的数据
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class SimpleSepsisDataset(Dataset):
    """
    简化的脓毒症数据集类，用于加载插补后的数据
    """
    def __init__(self, patient_data, vitals_data=None, labs_data=None, timeseries_data=None, id_col=None):
        """
        初始化数据集
        
        Args:
            patient_data: 包含患者信息的DataFrame
            vitals_data: 包含生命体征的DataFrame
            labs_data: 包含实验室检测值的DataFrame
            timeseries_data: 包含时间序列数据的DataFrame
            id_col: 用于标识患者的ID列名，如果提供则优先使用
        """
        self.patient_data = patient_data
        self.vitals_data = vitals_data
        self.labs_data = labs_data
        self.timeseries_data = timeseries_data
        self.id_col = id_col
        
        # 如果没有提供ID列，尝试查找
        if self.id_col is None:
            # 首先检查已知的ID列
            known_id_cols = ['subject_id', 'hadm_id', 'icustay_id']
            for col in known_id_cols:
                if col in self.patient_data.columns:
                    self.id_col = col
                    logger.info(f"自动选择 {col} 作为患者ID列")
                    break
                    
            # 如果找不到已知ID列，查找包含'id'的列
            if self.id_col is None:
                id_cols = [col for col in self.patient_data.columns if 'id' in col.lower()]
                if id_cols:
                    self.id_col = id_cols[0]
                    logger.info(f"找不到已知ID列，使用 {self.id_col} 作为患者ID列")
        
        # 预处理数据
        self._preprocess_data()
            
        logger.info(f"创建了SimpleSepsisDataset，包含{len(self.patient_ids)}名患者，"
                  f"{len(self.vitals)}条序列")
    
    def _validate_and_convert_numeric(self, df, cols=None):
        """
        验证并转换数据为有效的数值，替换NaN和无穷大
        
        Args:
            df: 要处理的DataFrame
            cols: 要处理的列，如果为None则处理所有数值列
            
        Returns:
            处理后的DataFrame
        """
        if df is None or df.empty:
            return df
        
        result = df.copy()
        
        # 确定要处理的列
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns
        
        # 替换无限值和NaN
        for col in cols:
            if col in result.columns:
                # 首先尝试转换非数值列
                if result[col].dtype == 'object':
                    try:
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                    except:
                        logger.warning(f"无法将列 {col} 转换为数值类型")
                
                # 如果是数值列，处理NaN和无穷大
                if pd.api.types.is_numeric_dtype(result[col]):
                    # 检测异常值
                    has_nan = result[col].isna().any()
                    has_inf = np.isinf(result[col].replace([np.inf, -np.inf], np.nan)).any()
                    
                    if has_nan or has_inf:
                        # 计算有效值的统计量用于替换
                        valid_values = result[col][~result[col].isna() & ~np.isinf(result[col])]
                        if not valid_values.empty:
                            median_val = valid_values.median()
                            std_val = valid_values.std()
                            # 如果标准差为0，使用一个小的值
                            if std_val == 0:
                                std_val = 0.01
                            
                            # 使用中位数替换NaN
                            result[col] = result[col].fillna(median_val)
                            
                            # 用中位数±3倍标准差替换±无穷大
                            result[col] = result[col].replace(np.inf, median_val + 3*std_val)
                            result[col] = result[col].replace(-np.inf, median_val - 3*std_val)
                            
                            logger.info(f"列 {col} 中的NaN和无穷大已替换为有效值")
                        else:
                            # 如果没有有效值，使用0
                            result[col] = result[col].fillna(0).replace([np.inf, -np.inf], 0)
                            logger.warning(f"列 {col} 没有有效值，使用0替换NaN和无穷大")
        
        return result
    
    def _preprocess_data(self):
        """简化的预处理，适用于已插补的数据"""
        # 提取唯一患者ID列表
        id_cols = [col for col in self.patient_data.columns if 'id' in col.lower()]
        if id_cols:
            self.id_col = id_cols[0]
            self.patient_ids = self.patient_data[self.id_col].unique()
        else:
            # 如果找不到ID列，使用索引
            self.id_col = None
            self.patient_ids = np.arange(len(self.patient_data))
        
        # 初始化数组来存储处理后的数据
        self.vitals = []  # 生命体征
        self.labs = []    # 实验室值
        self.drugs = []   # 药物使用
        self.labels = []  # 脓毒症标签
        self.time_indices = []  # 时间索引
        
        # 记录特征维度
        self.vitals_features = ['heart_rate', 'resp_rate', 'temperature', 'sbp', 'dbp', 'spo2']
        self.lab_features = ['wbc', 'creatinine', 'bun', 'lactate', 'platelet', 'bilirubin']
        self.drug_features = ['antibiotic', 'vasopressor']
        
        self.vitals_dim = len(self.vitals_features)
        self.lab_dim = len(self.lab_features)
        self.drug_dim = len(self.drug_features)
        
        # 预处理所有数据，确保数值有效
        self.patient_data = self._validate_and_convert_numeric(self.patient_data)
        self.vitals_data = self._validate_and_convert_numeric(self.vitals_data)
        self.labs_data = self._validate_and_convert_numeric(self.labs_data)
        self.timeseries_data = self._validate_and_convert_numeric(self.timeseries_data)
        
        # 如果有timeseries数据，优先使用
        if self.timeseries_data is not None and not self.timeseries_data.empty:
            self._process_from_timeseries()
            return
            
        # 否则尝试分别处理vitals和labs数据
        for patient_id in self.patient_ids:
            # 获取患者的标签
            if self.id_col:
                patient_info = self.patient_data[self.patient_data[self.id_col] == patient_id].iloc[0]
            else:
                patient_info = self.patient_data.iloc[patient_id]
            
            # 尝试获取sepsis标签
            sepsis_label = 0
            for col in patient_info.index:
                if 'sepsis' in col.lower() and 'label' in col.lower():
                    sepsis_label = patient_info[col]
                    break
            
            # 提取患者的生命体征数据
            vitals_mat = np.zeros((1, self.vitals_dim), dtype=np.float32)
            if self.vitals_data is not None and not self.vitals_data.empty and self.id_col in self.vitals_data.columns:
                patient_vitals = self.vitals_data[self.vitals_data[self.id_col] == patient_id]
                if not patient_vitals.empty:
                    # 提取可用的生命体征特征
                    available_cols = [col for col in self.vitals_features if col in patient_vitals.columns]
                    if available_cols:
                        for i, feature in enumerate(self.vitals_features):
                            if feature in available_cols:
                                # 使用均值代表该特征
                                vitals_mat[0, i] = patient_vitals[feature].mean()
            
            # 提取患者的实验室数据
            labs_mat = np.zeros((1, self.lab_dim), dtype=np.float32)
            if self.labs_data is not None and not self.labs_data.empty and self.id_col in self.labs_data.columns:
                patient_labs = self.labs_data[self.labs_data[self.id_col] == patient_id]
                if not patient_labs.empty:
                    # 提取可用的实验室特征
                    available_cols = [col for col in self.lab_features if col in patient_labs.columns]
                    if available_cols:
                        for i, feature in enumerate(self.lab_features):
                            if feature in available_cols:
                                # 使用均值代表该特征
                                labs_mat[0, i] = patient_labs[feature].mean()
            
            # 占位符药物数据
            drugs_mat = np.zeros((1, self.drug_dim), dtype=np.float32)
            
            # 标签和时间索引
            labels = np.array([sepsis_label], dtype=np.float32)
            time_indices = np.array([0], dtype=np.float32)
            
            # 确保所有矩阵都是有效数值
            vitals_mat = np.nan_to_num(vitals_mat, nan=0.0, posinf=1.0, neginf=-1.0)
            labs_mat = np.nan_to_num(labs_mat, nan=0.0, posinf=1.0, neginf=-1.0)
            drugs_mat = np.nan_to_num(drugs_mat, nan=0.0, posinf=1.0, neginf=-1.0)
            labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            time_indices = np.nan_to_num(time_indices, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 存储处理后的数据
            self.vitals.append(vitals_mat)
            self.labs.append(labs_mat)
            self.drugs.append(drugs_mat)
            self.labels.append(labels)
            self.time_indices.append(time_indices)
    
    def _process_from_timeseries(self):
        """从时间序列数据处理特征"""
        # 统计问题信息
        total_patients = len(self.patient_ids)
        zero_features_count = 0
        matched_features_count = 0
        
        # 记录可用列信息
        if not self.timeseries_data.empty:
            available_columns = set(self.timeseries_data.columns)
            logger.info(f"时间序列数据包含以下列: {sorted(list(available_columns))}")
            
            # 检查时间序列数据中的可能ID列
            ts_id_cols = [col for col in available_columns if any(id_name in col.lower() for id_name in 
                          ['subject_id', 'hadm_id', 'icustay_id', 'patient_id', 'patient', 'id'])]
            
            if ts_id_cols:
                logger.info(f"时间序列数据中找到的ID列: {ts_id_cols}")
            else:
                logger.warning("时间序列数据中没有找到任何ID列!")
            
            # 尝试识别患者数据中的ID列
            patient_id_cols = []
            if self.id_col:
                patient_id_cols.append(self.id_col)
            else:
                # 查找可能的ID列
                patient_id_cols = [col for col in self.patient_data.columns if any(id_name in col.lower() for id_name in 
                                  ['subject_id', 'hadm_id', 'icustay_id', 'patient_id', 'patient', 'id'])]
            
            if patient_id_cols:
                logger.info(f"患者数据中找到的ID列: {patient_id_cols}")
            else:
                logger.warning("患者数据中没有找到任何ID列!")
            
            # 创建所有可能的ID列对应关系用于匹配
            possible_id_mappings = []
            for p_col in patient_id_cols:
                for ts_col in ts_id_cols:
                    if p_col.lower() == ts_col.lower() or (
                        ('subject' in p_col.lower() and 'subject' in ts_col.lower()) or
                        ('hadm' in p_col.lower() and 'hadm' in ts_col.lower()) or
                        ('icustay' in p_col.lower() and 'icustay' in ts_col.lower()) or
                        ('patient' in p_col.lower() and 'patient' in ts_col.lower())
                    ):
                        possible_id_mappings.append((p_col, ts_col))
            
            if possible_id_mappings:
                logger.info(f"将尝试以下ID列进行匹配: {possible_id_mappings}")
            else:
                logger.warning("未找到可匹配的ID列!")
            
            # 为直接特征映射创建一个更简单的匹配方式
            # 这些是我们期望在time_series_sample.csv中找到的列名
            direct_features = {
                'vitals': {
                    'heart_rate': ['heart_rate', 'hr'],
                    'resp_rate': ['resp_rate', 'rr', 'respiratory_rate'],
                    'temperature': ['temperature', 'temp'],
                    'sbp': ['sbp', 'systolic'],
                    'dbp': ['dbp', 'diastolic'],
                    'spo2': ['spo2', 'o2sat']
                },
                'labs': {
                    'wbc': ['wbc'],
                    'creatinine': ['creatinine'],
                    'bun': ['bun'],
                    'lactate': ['lactate'],
                    'platelet': ['platelet'],
                    'bilirubin': ['bilirubin', 'bilirubin_total']
                },
                'drugs': {
                    'antibiotic': ['antibiotic'],
                    'vasopressor': ['vasopressor']
                }
            }
            
            # 尝试直接匹配特征列
            direct_column_matches = {}
            for category, features in direct_features.items():
                for feature, variants in features.items():
                    for variant in variants:
                        if variant in available_columns:
                            direct_column_matches[(category, feature)] = variant
                            logger.info(f"直接匹配到特征列: {category}.{feature} -> {variant}")
                            break
                        else:
                            # 尝试不太严格的匹配
                            matches = [col for col in available_columns if variant.lower() in col.lower()]
                            if matches:
                                direct_column_matches[(category, feature)] = matches[0]
                                logger.info(f"近似匹配到特征列: {category}.{feature} -> {matches[0]}")
                                break
        
        # 遍历每个患者进行特征提取
        for patient_id in self.patient_ids:
            # 获取患者的信息
            if self.id_col:
                patient_info = self.patient_data[self.patient_data[self.id_col] == patient_id].iloc[0]
                
                # 尝试使用多种ID列匹配时间序列数据
                patient_timeseries = pd.DataFrame()
                
                # 对于time_series_sample.csv，我们知道它使用subject_id作为ID
                if 'subject_id' in self.patient_data.columns and 'subject_id' in self.timeseries_data.columns:
                    subject_id = patient_info['subject_id']
                    patient_timeseries = self.timeseries_data[self.timeseries_data['subject_id'] == subject_id]
                    if not patient_timeseries.empty:
                        logger.info(f"通过subject_id={subject_id}匹配到患者时间序列数据: {len(patient_timeseries)}行")
                
                # 如果上面匹配失败，尝试使用hadm_id
                if patient_timeseries.empty and 'hadm_id' in self.patient_data.columns and 'hadm_id' in self.timeseries_data.columns:
                    hadm_id = patient_info['hadm_id']
                    patient_timeseries = self.timeseries_data[self.timeseries_data['hadm_id'] == hadm_id]
                    if not patient_timeseries.empty:
                        logger.info(f"通过hadm_id={hadm_id}匹配到患者时间序列数据: {len(patient_timeseries)}行")
                
                # 如果上面都失败，尝试使用icustay_id
                if patient_timeseries.empty and 'icustay_id' in self.patient_data.columns and 'icustay_id' in self.timeseries_data.columns:
                    icustay_id = patient_info['icustay_id']
                    patient_timeseries = self.timeseries_data[self.timeseries_data['icustay_id'] == icustay_id]
                    if not patient_timeseries.empty:
                        logger.info(f"通过icustay_id={icustay_id}匹配到患者时间序列数据: {len(patient_timeseries)}行")
                
                # 如果所有特定ID匹配都失败，回退到原始ID列匹配
                if patient_timeseries.empty:
                    patient_timeseries = self.timeseries_data[self.timeseries_data[self.id_col] == patient_id]
                    
                # 如果还是为空，尝试所有可能的ID映射
                if patient_timeseries.empty and 'possible_id_mappings' in locals():
                    for p_col, ts_col in possible_id_mappings:
                        if p_col in patient_info and not pd.isna(patient_info[p_col]):
                            id_value = patient_info[p_col]
                            patient_timeseries = self.timeseries_data[self.timeseries_data[ts_col] == id_value]
                            if not patient_timeseries.empty:
                                logger.info(f"通过匹配 {p_col}={id_value} 到 {ts_col} 找到患者时间序列数据: {len(patient_timeseries)}行")
                                break
            else:
                patient_info = self.patient_data.iloc[patient_id]
                # 没有ID列的情况下无法匹配时间序列
                patient_timeseries = pd.DataFrame()
            
            if patient_timeseries.empty:
                # 如果没有时间序列数据，创建占位符
                vitals_mat = np.zeros((1, self.vitals_dim), dtype=np.float32)
                labs_mat = np.zeros((1, self.lab_dim), dtype=np.float32)
                drugs_mat = np.zeros((1, self.drug_dim), dtype=np.float32)
                
                # 尝试从患者信息中获取标签
                sepsis_label = 0
                for col in patient_info.index:
                    if 'sepsis' in col.lower():
                        try:
                            sepsis_label = float(patient_info[col])
                            break
                        except:
                            pass
                
                labels = np.array([sepsis_label], dtype=np.float32)
                time_indices = np.array([0], dtype=np.float32)
                
                # 为没有时间序列数据的患者创建医学合理的默认值
                vitals_mat, labs_mat, drugs_mat = self._ensure_nonzero_features(vitals_mat, labs_mat, drugs_mat)
                
                zero_features_count += 1
            else:
                # 确保至少有一行数据 
                if len(patient_timeseries) < 1:
                    logger.warning(f"患者ID {patient_id} 的时间序列数据为空")
                    continue
                
                n_timepoints = len(patient_timeseries)
                logger.info(f"为患者ID {patient_id} 处理 {n_timepoints} 个时间点的数据")
                
                # 创建特征矩阵
                vitals_mat = np.zeros((n_timepoints, self.vitals_dim), dtype=np.float32)
                labs_mat = np.zeros((n_timepoints, self.lab_dim), dtype=np.float32)
                drugs_mat = np.zeros((n_timepoints, self.drug_dim), dtype=np.float32)
                
                # 首先尝试使用直接列名匹配 - 优先级最高
                features_found = False
                
                # 检查是否存在直接匹配的列
                if 'direct_column_matches' in locals():
                    # 处理生命体征特征
                    for i, feature in enumerate(self.vitals_features):
                        match_key = ('vitals', feature)
                        if match_key in direct_column_matches:
                            col = direct_column_matches[match_key]
                            try:
                                values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                vitals_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                features_found = True
                                logger.info(f"为患者 {patient_id} 直接匹配到特征: vitals.{feature} <- {col}")
                            except Exception as e:
                                logger.warning(f"处理列 '{col}' 时出错: {e}")
                
                    # 处理实验室值特征
                    for i, feature in enumerate(self.lab_features):
                        match_key = ('labs', feature)
                        if match_key in direct_column_matches:
                            col = direct_column_matches[match_key]
                            try:
                                values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                labs_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                features_found = True
                                logger.info(f"为患者 {patient_id} 直接匹配到特征: labs.{feature} <- {col}")
                            except Exception as e:
                                logger.warning(f"处理列 '{col}' 时出错: {e}")
                
                    # 处理药物特征
                    for i, feature in enumerate(self.drug_features):
                        match_key = ('drugs', feature)
                        if match_key in direct_column_matches:
                            col = direct_column_matches[match_key]
                            try:
                                values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                drugs_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                features_found = True
                                logger.info(f"为患者 {patient_id} 直接匹配到特征: drugs.{feature} <- {col}")
                            except Exception as e:
                                logger.warning(f"处理列 '{col}' 时出错: {e}")
                            break
                
                # 如果成功找到特征，记录
                if features_found:
                    matched_features_count += 1
                
                # 如果直接匹配失败，尝试使用备用方法（通用特征匹配）
                if not features_found:
                    logger.warning(f"患者ID {patient_id} 使用直接映射没有找到任何特征，尝试备用匹配方法")
                    
                    # 创建特征变体映射字典，显著增强匹配能力
                    feature_variants = {
                        # 生命体征特征变体
                        'heart_rate': ['heart_rate', 'hr', 'pulse', 'heart rate', 'heartrate', 'cardiac rate', 'pulse rate'],
                        'resp_rate': ['resp_rate', 'rr', 'resp rate', 'resprate', 'respiratory rate', 'breathing rate', 'respiration', 'respiratory'],
                        'temperature': ['temperature', 'temp', 'body temp', 'body temperature', 'fever', 'celsius', 'fahrenheit'],
                        'sbp': ['sbp', 'systolic', 'systolic bp', 'systolic blood pressure', 'bp_sys', 'sys_bp', 'systolic pressure'],
                        'dbp': ['dbp', 'diastolic', 'diastolic bp', 'diastolic blood pressure', 'bp_dia', 'dia_bp', 'diastolic pressure'],
                        'spo2': ['spo2', 'o2', 'o2 sat', 'oxygen', 'saturation', 'sat', 'oxygen saturation', 'o2sat'],
                        
                        # 实验室值特征变体
                        'wbc': ['wbc', 'white', 'leukocyte', 'white blood', 'white blood cell', 'white blood count', 'leukocytes'],
                        'creatinine': ['creat', 'cr', 'crea', 'creatinine', 'serum creatinine', 'kidney', 'renal function'],
                        'bun': ['bun', 'urea', 'nitrogen', 'blood urea', 'blood urea nitrogen', 'urea nitrogen'],
                        'lactate': ['lact', 'lac', 'lactate', 'lactic acid', 'blood lactate', 'serum lactate'],
                        'platelet': ['plat', 'plt', 'platelet', 'thrombocyte', 'platelet count', 'thrombocytes', 'platelets'],
                        'bilirubin': ['bili', 'bilirubin', 'tbili', 'total bilirubin', 'bilirubin total', 'liver', 'jaundice'],
                        
                        # 药物特征变体
                        'antibiotic': ['anti', 'antibiotic', 'abx', 'antimicrobial', 'antibiotics', 'abx use', 'antibiotic use'],
                        'vasopressor': ['vaso', 'pressor', 'vasopressor', 'norepi', 'epi', 'dopa', 'norepinephrine', 'epinephrine', 'dopamine']
                    }
                    
                    # 尝试使用更灵活的方法查找特征
                    for col in patient_timeseries.columns:
                        col_lower = col.lower()
                        
                        # 生命体征特征
                        for i, feature in enumerate(self.vitals_features):
                            for variant in feature_variants.get(feature, [feature]):
                                if variant.lower() in col_lower:
                                    try:
                                        values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                        vitals_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                        features_found = True
                                        logger.info(f"备用方法: 列 '{col}' 映射到生命体征 '{feature}'")
                                    except Exception as e:
                                        logger.warning(f"处理列 '{col}' 时出错: {e}")
                                    break
                        
                        # 实验室值特征
                        for i, feature in enumerate(self.lab_features):
                            for variant in feature_variants.get(feature, [feature]):
                                if variant.lower() in col_lower:
                                    try:
                                        values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                        labs_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                        features_found = True
                                        logger.info(f"备用方法: 列 '{col}' 映射到实验室值 '{feature}'")
                                    except Exception as e:
                                        logger.warning(f"处理列 '{col}' 时出错: {e}")
                                    break
                        
                        # 药物使用特征
                        for i, feature in enumerate(self.drug_features):
                            for variant in feature_variants.get(feature, [feature]):
                                if variant.lower() in col_lower:
                                    try:
                                        values = pd.to_numeric(patient_timeseries[col], errors='coerce').values
                                        drugs_mat[:, i] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                                        features_found = True
                                        logger.info(f"备用方法: 列 '{col}' 映射到药物 '{feature}'")
                                    except Exception as e:
                                        logger.warning(f"处理列 '{col}' 时出错: {e}")
                                    break
                    
                    # 如果成功找到特征，记录
                    if features_found:
                        matched_features_count += 1
                
                # 提取标签
                sepsis_label = 0
                label_cols = [col for col in patient_timeseries.columns if 'sepsis' in col.lower() or 'label' in col.lower()]
                if label_cols:
                    try:
                        labels = patient_timeseries[label_cols[0]].astype(np.float32).values
                        labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
                    except Exception as e:
                        logger.warning(f"无法处理标签 {label_cols[0]}: {e}")
                        # 从患者信息中获取标签
                        for col in patient_info.index:
                            if 'sepsis' in col.lower():
                                try:
                                    sepsis_label = float(patient_info[col])
                                    break
                                except:
                                    pass
                        labels = np.full(n_timepoints, sepsis_label, dtype=np.float32)
                else:
                    # 从患者信息中获取标签
                    for col in patient_info.index:
                        if 'sepsis' in col.lower():
                            try:
                                sepsis_label = float(patient_info[col])
                                break
                            except:
                                pass
                    labels = np.full(n_timepoints, sepsis_label, dtype=np.float32)
                
                # 提取时间索引
                time_cols = [col for col in patient_timeseries.columns if 'time' in col.lower() or 'hour' in col.lower()]
                if time_cols:
                    try:
                        time_values = patient_timeseries[time_cols[0]].values
                        
                        # 确保时间值是数值类型
                        if time_values.dtype == np.dtype('O'):
                            time_indices = np.array([float(x) if x is not None and not pd.isna(x) else 0.0 
                                                    for x in time_values], dtype=np.float32)
                        else:
                            time_indices = time_values.astype(np.float32)
                        
                            time_indices = np.nan_to_num(time_indices, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        logger.warning(f"无法将列 {time_cols[0]} 转换为浮点数: {e}")
                        time_indices = np.arange(n_timepoints, dtype=np.float32)
                else:
                    time_indices = np.arange(n_timepoints, dtype=np.float32)
            
                # 对全零样本应用医学合理的默认值
                all_zero = (np.abs(vitals_mat).sum() < 1e-6) and (np.abs(labs_mat).sum() < 1e-6) and (np.abs(drugs_mat).sum() < 1e-6)
                if all_zero:
                    # 如果应该有特征但全零，这是个异常情况，记录更详细信息
                    if features_found:
                        logger.warning(f"患者ID {patient_id} 已找到特征映射但矩阵仍然全零，这可能是数据问题")
                        logger.warning(f"患者时间序列数据样本: \n{patient_timeseries.head(1).to_string()}")
                    
                    # 应用_ensure_nonzero_features并确保使用其返回值
                    vitals_mat, labs_mat, drugs_mat = self._ensure_nonzero_features(vitals_mat, labs_mat, drugs_mat)
                    zero_features_count += 1
            
            # 存储处理后的数据
            self.vitals.append(vitals_mat)
            self.labs.append(labs_mat)
            self.drugs.append(drugs_mat)
            self.labels.append(labels)
            self.time_indices.append(time_indices)
        
        # 报告特征匹配统计
        logger.info(f"特征提取完成: 总患者数 {total_patients}, 成功匹配特征 {matched_features_count}, 全零样本 {zero_features_count}")
        
        # 如果所有样本都是全零，发出严重警告
        if zero_features_count == total_patients:
            logger.error("严重警告: 所有样本都是全零特征! 请检查数据格式和列名是否匹配预期!")
            
            # 打印数据示例辅助调试
            if not self.timeseries_data.empty:
                logger.info(f"时间序列数据前5行:\n{self.timeseries_data.head().to_string()}")
                
                # 检查主要特征列是否有非零值
                for possible_feature in ['heart_rate', 'resp_rate', 'temperature', 'wbc', 'antibiotic']:
                    matching_cols = [col for col in self.timeseries_data.columns if possible_feature.lower() in col.lower()]
                    if matching_cols:
                        for col in matching_cols:
                            has_nonzero = (self.timeseries_data[col].abs() > 1e-6).any()
                            logger.info(f"列 '{col}' 是否有非零值: {has_nonzero}")
                            if has_nonzero:
                                nonzero_count = (self.timeseries_data[col].abs() > 1e-6).sum()
                                logger.info(f"列 '{col}' 有 {nonzero_count} 个非零值，示例值: {self.timeseries_data[col].dropna().iloc[:5].tolist()}")
    
    def _ensure_nonzero_features(self, vitals_mat, labs_mat, drugs_mat):
        """
        确保每个样本至少有一个非零特征，防止被完全掩码
        使用医学上合理的默认值替代随机值，并引入阈值检测
        """
        # 设置零值容忍阈值 - 如果总和小于此值，视为需要填充
        zero_threshold = 1e-6
        
        # 检查是否为全零矩阵
        all_zero = (np.abs(vitals_mat.sum()) < zero_threshold and 
                    np.abs(labs_mat.sum()) < zero_threshold and 
                    np.abs(drugs_mat.sum()) < zero_threshold)
                    
        # 记录特征矩阵的统计信息
        logger.debug(f"特征矩阵统计 - 生命体征: 形状={vitals_mat.shape}, 总和={vitals_mat.sum():.4f}, "
                   f"实验室值: 形状={labs_mat.shape}, 总和={labs_mat.sum():.4f}, "
                   f"药物: 形状={drugs_mat.shape}, 总和={drugs_mat.sum():.4f}")
        
        if all_zero:
            
            # 所有特征都接近零，填充医学上合理的默认值
            n_timepoints = vitals_mat.shape[0]
            
            # 为生命体征添加合理的默认值
            if vitals_mat.shape[1] >= 1:  # 心率 (HR)
                vitals_mat[:, 0] = 75 + np.random.normal(0, 5, size=n_timepoints)  # 正常心率约75次/分钟
            if vitals_mat.shape[1] >= 2:  # 呼吸频率 (RR)
                vitals_mat[:, 1] = 16 + np.random.normal(0, 2, size=n_timepoints)  # 正常呼吸频率约16次/分钟
            if vitals_mat.shape[1] >= 3:  # 体温 (Temperature)
                vitals_mat[:, 2] = 36.8 + np.random.normal(0, 0.3, size=n_timepoints)  # 正常体温约36.8°C
            if vitals_mat.shape[1] >= 4:  # 收缩压 (SBP)
                vitals_mat[:, 3] = 120 + np.random.normal(0, 5, size=n_timepoints)  # 正常收缩压约120 mmHg
            if vitals_mat.shape[1] >= 5:  # 舒张压 (DBP)
                vitals_mat[:, 4] = 80 + np.random.normal(0, 5, size=n_timepoints)  # 正常舒张压约80 mmHg
            if vitals_mat.shape[1] >= 6:  # 血氧饱和度 (SpO2)
                vitals_mat[:, 5] = 98 + np.random.normal(0, 1, size=n_timepoints)  # 正常血氧饱和度约98%
                # 确保血氧饱和度不超过100%
                vitals_mat[:, 5] = np.clip(vitals_mat[:, 5], 90, 100)
            
            # 为实验室值添加合理的默认值
            if labs_mat.shape[1] >= 1:  # 白细胞计数 (WBC)
                labs_mat[:, 0] = 7.5 + np.random.normal(0, 1.5, size=n_timepoints)  # 正常白细胞计数约7.5×10^9/L
            if labs_mat.shape[1] >= 2:  # 肌酐 (Creatinine)
                labs_mat[:, 1] = 0.9 + np.random.normal(0, 0.2, size=n_timepoints)  # 正常肌酐约0.9 mg/dL
                # 确保肌酐不小于0
                labs_mat[:, 1] = np.maximum(0.5, labs_mat[:, 1])
            if labs_mat.shape[1] >= 3:  # 血尿素氮 (BUN)
                labs_mat[:, 2] = 14 + np.random.normal(0, 3, size=n_timepoints)  # 正常BUN约14 mg/dL
            if labs_mat.shape[1] >= 4:  # 乳酸 (Lactate)
                labs_mat[:, 3] = 1.2 + np.random.normal(0, 0.3, size=n_timepoints)  # 正常乳酸约1.2 mmol/L
                # 确保乳酸不小于0
                labs_mat[:, 3] = np.maximum(0.5, labs_mat[:, 3])
            if labs_mat.shape[1] >= 5:  # 血小板计数 (Platelet)
                labs_mat[:, 4] = 250 + np.random.normal(0, 40, size=n_timepoints)  # 正常血小板约250×10^9/L
            if labs_mat.shape[1] >= 6:  # 总胆红素 (Bilirubin)
                labs_mat[:, 5] = 0.8 + np.random.normal(0, 0.2, size=n_timepoints)  # 正常总胆红素约0.8 mg/dL
                # 确保胆红素不小于0
                labs_mat[:, 5] = np.maximum(0.2, labs_mat[:, 5])
            
            # 药物使用默认值 - 假设大多数患者一开始不使用这些药物
            if drugs_mat.shape[1] >= 1:  # 抗生素
                drugs_mat[:, 0] = np.random.choice([0, 1], size=n_timepoints, p=[0.8, 0.2])  # 20%概率使用抗生素
            if drugs_mat.shape[1] >= 2:  # 升压药
                drugs_mat[:, 1] = np.random.choice([0, 1], size=n_timepoints, p=[0.9, 0.1])  # 10%概率使用升压药
            
            # 对于时间序列数据，可以添加一些时间相关的变化
            if n_timepoints > 1:
                # 对于病程中期，为少数时间点添加药物使用
                # 病程中间部分偶尔有药物使用
                middle_start = n_timepoints//4
                middle_end = 3*n_timepoints//4
                if middle_start < middle_end and drugs_mat.shape[1] >= 1:
                    # 约20%的概率使用抗生素，在中期有30%的概率
                    drug_mask = np.random.random(size=middle_end-middle_start) < 0.3
                    if drug_mask.any():
                        drugs_mat[middle_start:middle_end, 0][drug_mask] = 1
            
            # 记录填充后的统计信息
            logger.debug(f"填充后特征矩阵统计 - 生命体征: 总和={vitals_mat.sum():.4f}, "
                       f"实验室值: 总和={labs_mat.sum():.4f}, "
                       f"药物: 总和={drugs_mat.sum():.4f}")
            
            
            # 强制返回修改后的矩阵
            return vitals_mat, labs_mat, drugs_mat
        
        # 如果不是全零，返回原始矩阵
        return vitals_mat, labs_mat, drugs_mat
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.vitals)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            dict: 包含各种特征和标签的字典
        """
        # 获取对应的数据
        vitals = self.vitals[idx]
        labs = self.labs[idx]
        drugs = self.drugs[idx]
        labels = self.labels[idx]
        time_indices = self.time_indices[idx]
        
        # 确保所有数据都是有效的数值
        vitals = np.nan_to_num(vitals, nan=0.0, posinf=1.0, neginf=-1.0)
        labs = np.nan_to_num(labs, nan=0.0, posinf=1.0, neginf=-1.0)
        drugs = np.nan_to_num(drugs, nan=0.0, posinf=1.0, neginf=-1.0)
        labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        time_indices = np.nan_to_num(time_indices, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为PyTorch张量
        vitals_tensor = torch.FloatTensor(vitals)
        labs_tensor = torch.FloatTensor(labs)
        drugs_tensor = torch.FloatTensor(drugs)
        labels_tensor = torch.FloatTensor(labels)
        time_indices_tensor = torch.FloatTensor(time_indices)
        
        return {
            'vitals': vitals_tensor,
            'labs': labs_tensor,
            'drugs': drugs_tensor,
            'labels': labels_tensor,
            'time_indices': time_indices_tensor,
        }

def collate_fn(batch):
    """
    自定义批处理函数
    处理不同长度的序列
    """
    # 按键分组
    vitals = [item['vitals'] for item in batch]
    labs = [item['labs'] for item in batch]
    drugs = [item['drugs'] for item in batch]
    labels = [item['labels'] for item in batch]
    time_indices = [item['time_indices'] for item in batch]
    
    # 获取批次中每个序列的长度
    lengths = torch.tensor([v.size(0) for v in vitals])
    
    # 获取最大长度
    max_len = max(lengths).item()
    
    # 创建填充掩码
    batch_size = len(batch)
    attention_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    # 填充序列
    padded_vitals = torch.zeros(batch_size, max_len, vitals[0].size(1))
    padded_labs = torch.zeros(batch_size, max_len, labs[0].size(1))
    padded_drugs = torch.zeros(batch_size, max_len, drugs[0].size(1))
    padded_labels = torch.zeros(batch_size, max_len)
    padded_time_indices = torch.zeros(batch_size, max_len)
    
    # 记录统计信息
    zero_features_count = 0
    
    # 填充每个序列
    for i, (v, l, d, y, t, length) in enumerate(zip(vitals, labs, drugs, labels, time_indices, lengths)):
        padded_vitals[i, :length] = v
        padded_labs[i, :length] = l
        padded_drugs[i, :length] = d
        padded_labels[i, :length] = y
        padded_time_indices[i, :length] = t
        
        # 检查是否全零
        is_zero_feature = (torch.abs(v).sum() < 1e-6) and (torch.abs(l).sum() < 1e-6) and (torch.abs(d).sum() < 1e-6)
        if is_zero_feature:
            zero_features_count += 1
        
        # 标记填充部分
        attention_mask[i, length:] = False
    
    # 最终检查确保没有NaN或Inf
    padded_vitals = torch.nan_to_num(padded_vitals, nan=0.0, posinf=1.0, neginf=-1.0)
    padded_labs = torch.nan_to_num(padded_labs, nan=0.0, posinf=1.0, neginf=-1.0)
    padded_drugs = torch.nan_to_num(padded_drugs, nan=0.0, posinf=1.0, neginf=-1.0)
    padded_labels = torch.nan_to_num(padded_labels, nan=0.0, posinf=1.0, neginf=0.0)
    padded_time_indices = torch.nan_to_num(padded_time_indices, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 报告批次中的全零样本统计
    if zero_features_count > 0:
        logger.warning(f"批次中存在 {zero_features_count}/{batch_size} 个全零样本")
    
    # 确保每个样本至少有一个非填充位置有有效特征
    ensure_valid_features_in_batch(padded_vitals, padded_labs, padded_drugs, attention_mask)
    
    return {
        'vitals': padded_vitals,
        'labs': padded_labs,
        'drugs': padded_drugs,
        'labels': padded_labels,
        'time_indices': padded_time_indices,
        'attention_mask': attention_mask,
        'lengths': lengths
    }

def ensure_valid_features_in_batch(vitals, labs, drugs, attention_mask):
    """
    确保批次中每个样本在有效位置上至少有一个非零特征
    使用医学上合理的默认值替代随机值
    
    Args:
        vitals: 生命体征特征张量 [batch_size, seq_len, vitals_dim]
        labs: 实验室值特征张量 [batch_size, seq_len, lab_dim]
        drugs: 药物使用特征张量 [batch_size, seq_len, drug_dim]
        attention_mask: 注意力掩码 [batch_size, seq_len]，False表示非填充位置
    """
    batch_size = vitals.size(0)
    zero_threshold = 1e-6
    fixed_samples = 0
    
    for i in range(batch_size):
        # 获取非填充位置的掩码
        non_padding = ~attention_mask[i]
        
        if non_padding.any():
            # 检查非填充位置的特征是否全为零
            non_padding_vitals = torch.abs(vitals[i, non_padding].sum())
            non_padding_labs = torch.abs(labs[i, non_padding].sum())
            non_padding_drugs = torch.abs(drugs[i, non_padding].sum())
            
            # 详细记录样本的特征状态
            logger.debug(f"样本 {i} 非填充位置统计: 生命体征={non_padding_vitals:.4f}, "
                       f"实验室值={non_padding_labs:.4f}, "
                       f"药物={non_padding_drugs:.4f}")
            
            if non_padding_vitals < zero_threshold and non_padding_labs < zero_threshold and non_padding_drugs < zero_threshold:
                # 所有特征都接近零，在非填充位置添加医学上合理的默认值
                valid_positions = torch.where(non_padding)[0]
                n_valid = len(valid_positions)
                
                if n_valid > 0:
                    fixed_samples += 1
                    
                    # 为所有有效位置都添加合理值，而不仅仅是第一个
                    for pos_idx, pos in enumerate(valid_positions):
                        time_factor = pos_idx / max(1, (n_valid - 1))  # 时间进程因子 (0-1)
                        
                        # 为生命体征添加合理的默认值
                        # 心率 (HR) - 随时间略微上升
                        base_hr = 75 + torch.randn(1, device=vitals.device) * 5
                        vitals[i, pos, 0] = base_hr * (1 + 0.1 * time_factor)
                        
                        # 呼吸频率 (RR) - 随时间略微上升
                        if vitals.size(2) > 1:
                            base_rr = 16 + torch.randn(1, device=vitals.device) * 2
                            vitals[i, pos, 1] = base_rr * (1 + 0.15 * time_factor)
                        
                        # 体温 (Temperature) - 可能略微上升
                        if vitals.size(2) > 2:
                            base_temp = 36.8 + torch.randn(1, device=vitals.device) * 0.3
                            vitals[i, pos, 2] = base_temp * (1 + 0.02 * time_factor)
                        
                        # 收缩压 (SBP) - 可能有波动
                        if vitals.size(2) > 3:
                            base_sbp = 120 + torch.randn(1, device=vitals.device) * 5
                            vitals[i, pos, 3] = base_sbp * (1 + 0.05 * (torch.sin(torch.tensor(pos_idx * 0.5))))
                        
                        # 舒张压 (DBP) - 与收缩压协同变化
                        if vitals.size(2) > 4:
                            base_dbp = 80 + torch.randn(1, device=vitals.device) * 5
                            vitals[i, pos, 4] = base_dbp * (1 + 0.05 * (torch.sin(torch.tensor(pos_idx * 0.5))))
                        
                        # 血氧饱和度 (SpO2) - 可能略微下降
                        if vitals.size(2) > 5:
                            base_spo2 = 98 + torch.randn(1, device=vitals.device)
                            vitals[i, pos, 5] = base_spo2 * (1 - 0.02 * time_factor)
                            # 确保血氧饱和度不超过100%且不低于90%
                            vitals[i, pos, 5] = torch.clamp(vitals[i, pos, 5], 90, 100)
                        
                        # 为实验室值添加合理的默认值
                        # 白细胞计数 (WBC) - 可能上升
                        base_wbc = 7.5 + torch.randn(1, device=labs.device) * 1.5
                        labs[i, pos, 0] = base_wbc * (1 + 0.2 * time_factor)
                        
                        # 肌酐 (Creatinine) - 可能上升
                        if labs.size(2) > 1:
                            base_creat = 0.9 + torch.randn(1, device=labs.device) * 0.2
                            labs[i, pos, 1] = base_creat * (1 + 0.3 * time_factor)
                            # 确保肌酐不小于0
                            labs[i, pos, 1] = torch.max(torch.tensor(0.5, device=labs.device), labs[i, pos, 1])
                        
                        # 血尿素氮 (BUN) - 可能上升
                        if labs.size(2) > 2:
                            base_bun = 14 + torch.randn(1, device=labs.device) * 3
                            labs[i, pos, 2] = base_bun * (1 + 0.25 * time_factor)
                        
                        # 乳酸 (Lactate) - 可能上升
                        if labs.size(2) > 3:
                            base_lact = 1.2 + torch.randn(1, device=labs.device) * 0.3
                            labs[i, pos, 3] = base_lact * (1 + 0.4 * time_factor)
                            # 确保乳酸不小于0
                            labs[i, pos, 3] = torch.max(torch.tensor(0.5, device=labs.device), labs[i, pos, 3])
                        
                        # 血小板计数 (Platelet) - 可能下降
                        if labs.size(2) > 4:
                            base_plt = 250 + torch.randn(1, device=labs.device) * 40
                            labs[i, pos, 4] = base_plt * (1 - 0.2 * time_factor)
                        
                        # 总胆红素 (Bilirubin) - 可能上升
                        if labs.size(2) > 5:
                            base_bili = 0.8 + torch.randn(1, device=labs.device) * 0.2
                            labs[i, pos, 5] = base_bili * (1 + 0.3 * time_factor)
                            # 确保胆红素不小于0
                            labs[i, pos, 5] = torch.max(torch.tensor(0.2, device=labs.device), labs[i, pos, 5])
                        
                        # 药物使用 - 在后期时间点可能开始使用
                        if drugs.size(2) >= 1:  # 抗生素
                            # 时间越晚，使用抗生素的概率越高
                            use_prob = 0.1 + 0.4 * time_factor
                            drugs[i, pos, 0] = 1.0 if torch.rand(1, device=drugs.device) < use_prob else 0.0
                        
                        if drugs.size(2) >= 2:  # 升压药
                            # 时间越晚，使用升压药的概率越高，但总体低于抗生素
                            use_prob = 0.05 + 0.25 * time_factor
                            drugs[i, pos, 1] = 1.0 if torch.rand(1, device=drugs.device) < use_prob else 0.0
    
    if fixed_samples > 0:
        logger.info(f"在批次中共修复了 {fixed_samples}/{batch_size} 个全零样本")

def load_data(data_dir, batch_size=32, max_samples=None):
    """
    加载数据
    
    Args:
        data_dir: 数据目录
        batch_size: 批处理大小
        max_samples: 最大样本数量
        
    Returns:
        tuple: (data_loaders, feature_dims)
    """
    logger.info(f"从{data_dir}加载数据...")
    
    # 查找可能的数据文件
    possible_patient_files = ['sampled_patients.csv', 'all_patients.csv', 'patients_imputed.csv', 'sepsis_patients.csv', 'all_patients_limited.csv']
    possible_vitals_files = ['vital_signs.csv', 'vital_signs_interim_5.csv', 'vitals_imputed.csv']
    possible_labs_files = ['lab_values.csv', 'lab_values_interim_4.csv', 'labs_imputed.csv', 'lab_values_sample.csv']
    possible_timeseries_files = ['timeseries.csv', 'time_series.csv', 'timeseries_imputed.csv', 'time_series_sample.csv']
    
    # 查找患者数据文件
    patients_file = None
    for file_name in possible_patient_files:
        temp_file = os.path.join(data_dir, file_name)
        if os.path.exists(temp_file):
            patients_file = temp_file
            logger.info(f"找到患者数据文件: {file_name}")
            break
    
    if patients_file is None:
        raise FileNotFoundError(f"在{data_dir}中未找到患者数据文件")
    
    # 加载患者数据
    patient_data = pd.read_csv(patients_file)
    logger.info(f"加载了患者数据: {len(patient_data)}行")
    
    # 检查患者数据中的ID列
    expected_id_cols = ['subject_id', 'hadm_id', 'icustay_id']
    patient_id_cols = [col for col in expected_id_cols if col in patient_data.columns]
    
    if not patient_id_cols:
        # 尝试查找其他可能的ID列
        id_cols = [col for col in patient_data.columns if 'id' in col.lower()]
        patient_id_cols = id_cols if id_cols else []
    
    if patient_id_cols:
        # 选择主ID列作为患者标识符
        primary_id_col = patient_id_cols[0]
        logger.info(f"使用 {primary_id_col} 作为主要患者ID列")
        
        # 检查ID列中的值
        logger.info(f"患者ID范例: {patient_data[primary_id_col].head(3).tolist()}")
        logger.info(f"患者ID列中的唯一值数量: {patient_data[primary_id_col].nunique()}")
        
        # 检查患者数据是否存在重复ID
        if patient_data[primary_id_col].duplicated().any():
            logger.warning(f"患者数据中存在重复的 {primary_id_col} 值!")
            
            # 如果有多个ID列，尝试使用组合ID
            if len(patient_id_cols) > 1:
                logger.info(f"尝试使用组合ID: {patient_id_cols}")
                # 检查组合ID是否唯一
                if patient_data.duplicated(patient_id_cols).any():
                    logger.warning(f"即使使用组合ID {patient_id_cols}，患者数据中仍存在重复!")
                else:
                    logger.info(f"使用组合ID {patient_id_cols} 可以唯一标识患者")
    else:
        logger.warning("未找到患者ID列，将使用索引作为ID")
        primary_id_col = None
    
    # 如果指定了最大样本数
    if max_samples and len(patient_data) > max_samples:
        patient_data = patient_data.sample(max_samples, random_state=42)
        logger.info(f"限制样本数为{max_samples}")
    
    # 查找时间序列数据文件 - 优先处理time_series_sample.csv
    timeseries_file = os.path.join(data_dir, 'time_series_sample.csv')
    if os.path.exists(timeseries_file):
        logger.info(f"找到目标时间序列数据文件: time_series_sample.csv")
    else:
        # 如果找不到，尝试其他可能的时间序列文件
        timeseries_file = None
        for file_name in possible_timeseries_files:
            temp_file = os.path.join(data_dir, file_name)
            if os.path.exists(temp_file):
                timeseries_file = temp_file
                logger.info(f"找到时间序列数据文件: {file_name}")
                break
    
    # 根据时间序列数据文件的存在情况确定加载策略
    vitals_data = pd.DataFrame()
    labs_data = pd.DataFrame() 
    timeseries_data = pd.DataFrame()
    
    if timeseries_file:
        # 加载时间序列数据
        timeseries_data = pd.read_csv(timeseries_file)
        logger.info(f"加载了时间序列数据: {len(timeseries_data)}行")
        
        # 检查时间序列数据中的ID列
        expected_ts_id_cols = ['subject_id', 'hadm_id', 'icustay_id']
        ts_id_cols = [col for col in expected_ts_id_cols if col in timeseries_data.columns]
        
        if not ts_id_cols:
            # 尝试查找其他可能的ID列
            ts_id_cols = [col for col in timeseries_data.columns if 'id' in col.lower()]
        
        if ts_id_cols:
            logger.info(f"时间序列数据中的ID列: {ts_id_cols}")
            
            # 分析患者ID匹配情况
            if patient_id_cols:
                # 对于每个可能的ID列对，检查匹配情况
                best_match_count = 0
                best_match_pair = None
                
                for p_col in patient_id_cols:
                    for ts_col in ts_id_cols:
                        if p_col == ts_col:  # 优先考虑同名列
                            patient_ids = set(patient_data[p_col])
                            ts_ids = set(timeseries_data[ts_col])
                            common_ids = patient_ids.intersection(ts_ids)
                            
                            if len(common_ids) > best_match_count:
                                best_match_count = len(common_ids)
                                best_match_pair = (p_col, ts_col)
                                
                                logger.info(f"ID列对 {p_col} <-> {ts_col} 匹配到 {len(common_ids)} 个患者")
                                
                                if len(common_ids) == len(patient_ids):
                                    logger.info(f"完美匹配! 所有患者都有对应的时间序列数据")
                                    break
                
                if best_match_pair:
                    p_col, ts_col = best_match_pair
                    logger.info(f"选择最佳匹配的ID列对: {p_col} <-> {ts_col}")
                    
                    # 设置为主要ID列
                    primary_id_col = p_col
                    
                    # 如果匹配率过低，发出警告
                    patient_ids = set(patient_data[p_col])
                    match_ratio = best_match_count / len(patient_ids)
                    if match_ratio < 0.5:
                        logger.warning(f"警告: 超过一半的患者没有对应的时间序列数据! 匹配率: {match_ratio*100:.2f}%")
                        
                        # 检查有多少患者在时间序列数据中没有记录
                        missing_ids = patient_ids - set(timeseries_data[ts_col])
                        logger.warning(f"共有 {len(missing_ids)} 个患者在时间序列数据中没有记录")
                        if len(missing_ids) <= 10:
                            logger.warning(f"缺失的患者ID: {list(missing_ids)}")
                else:
                    logger.warning("无法找到有效的ID列对进行匹配!")
        
        # 检查time_series_sample.csv中的特征列
        expected_features = {
            'vitals': ['heart_rate', 'resp_rate', 'temperature', 'sbp', 'dbp', 'spo2'],
            'labs': ['wbc', 'creatinine', 'bun', 'lactate', 'platelet', 'bilirubin_total'],
            'drugs': ['antibiotic', 'vasopressor']
        }
        
        # 检查每个特征类别中的列是否存在
        feature_columns_found = {}
        for category, features in expected_features.items():
            found = []
            for feature in features:
                if feature in timeseries_data.columns:
                    found.append(feature)
                    feature_columns_found[feature] = feature
                else:
                    # 尝试模糊匹配
                    matches = [col for col in timeseries_data.columns if feature.lower() in col.lower()]
                    if matches:
                        found.append(matches[0])
                        feature_columns_found[feature] = matches[0]
            
            logger.info(f"在时间序列数据中找到的{category}特征: {found}/{len(features)}")
        
        # 检查数据中是否有重要的统计量
        for category, features in expected_features.items():
            for feature in features:
                mapped_col = feature_columns_found.get(feature, None)
                if mapped_col and mapped_col in timeseries_data.columns:
                    non_na_count = timeseries_data[mapped_col].notna().sum()
                    if non_na_count > 0:
                        non_zero_count = (timeseries_data[mapped_col] != 0).sum()
                        mean_val = timeseries_data[mapped_col].mean()
                        logger.info(f"特征 {mapped_col}: 非缺失值 {non_na_count}, 非零值 {non_zero_count}, 平均值 {mean_val:.4f}")
                    else:
                        logger.warning(f"特征 {mapped_col} 全为缺失值!")
    else:
        # 回退到分别加载生命体征和实验室数据
        logger.warning("未找到时间序列数据文件，将尝试分别加载生命体征和实验室数据")
    
    # 查找生命体征数据文件
    vitals_file = None
    for file_name in possible_vitals_files:
        temp_file = os.path.join(data_dir, file_name)
        if os.path.exists(temp_file):
            vitals_file = temp_file
            logger.info(f"找到生命体征数据文件: {file_name}")
            break
    
    # 修复加载生命体征数据部分
    if vitals_file:
        vitals_data = pd.read_csv(vitals_file)
        logger.info(f"加载了生命体征数据: {len(vitals_data)}行")
        logger.info(f"生命体征数据列名: {list(vitals_data.columns)}")
    
    # 查找实验室值数据文件
    labs_file = None
    for file_name in possible_labs_files:
        temp_file = os.path.join(data_dir, file_name)
        if os.path.exists(temp_file):
            labs_file = temp_file
            logger.info(f"找到实验室值数据文件: {file_name}")
            break
    
    # 修复实验室数据部分
    if labs_file:
        labs_data = pd.read_csv(labs_file)
        logger.info(f"加载了实验室值数据: {len(labs_data)}行")
        logger.info(f"实验室值数据列名: {list(labs_data.columns)}")
    
    # 确保ID列被正确设置
    id_col = primary_id_col
    logger.info(f"最终选定的ID列: {id_col}")
    
    # 如果有时间序列数据，直接使用它创建数据集
    if not timeseries_data.empty:
        logger.info("使用时间序列数据创建数据集")
    
    # 分割数据集
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
        # 首先随机打乱数据 - 确保按照患者ID分组
        if id_col and id_col in patient_data.columns:
            # 获取唯一患者ID
            unique_patients = patient_data[id_col].unique()
            np.random.seed(42)
            np.random.shuffle(unique_patients)
            
            # 计算分割点
            n_patients = len(unique_patients)
            train_end = int(n_patients * train_ratio)
            val_end = train_end + int(n_patients * val_ratio)
            
            # 分割患者ID
            train_patient_ids = unique_patients[:train_end]
            val_patient_ids = unique_patients[train_end:val_end]
            test_patient_ids = unique_patients[val_end:]
            
            # 基于ID分割患者数据
            train_patients = patient_data[patient_data[id_col].isin(train_patient_ids)]
            val_patients = patient_data[patient_data[id_col].isin(val_patient_ids)]
            test_patients = patient_data[patient_data[id_col].isin(test_patient_ids)]
            
            logger.info(f"数据分割完成。训练集: {len(train_patient_ids)}名患者, "
                      f"验证集: {len(val_patient_ids)}名患者, "
                      f"测试集: {len(test_patient_ids)}名患者")
        else:
            # 如果没有ID列，直接分割数据
            patient_data = patient_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算分割点
    n_samples = len(patient_data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # 分割患者数据
    train_patients = patient_data.iloc[:train_end]
    val_patients = patient_data.iloc[train_end:val_end]
    test_patients = patient_data.iloc[val_end:]
    
    logger.info(f"数据分割完成。训练集: {len(train_patients)}名患者, "
               f"验证集: {len(val_patients)}名患者, "
               f"测试集: {len(test_patients)}名患者")
    
        # 创建数据集，明确传递ID列名
    train_dataset = SimpleSepsisDataset(train_patients, id_col=id_col, timeseries_data=timeseries_data)
    val_dataset = SimpleSepsisDataset(val_patients, id_col=id_col, timeseries_data=timeseries_data)
    test_dataset = SimpleSepsisDataset(test_patients, id_col=id_col, timeseries_data=timeseries_data)
    else:
        # 如果没有时间序列数据，使用分开的生命体征和实验室数据
        # 此部分代码保持不变...
    train_dataset = SimpleSepsisDataset(train_patients, vitals_data, labs_data, timeseries_data)
    val_dataset = SimpleSepsisDataset(val_patients, vitals_data, labs_data, timeseries_data)
    test_dataset = SimpleSepsisDataset(test_patients, vitals_data, labs_data, timeseries_data)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 创建数据加载器字典
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # 特征维度
    feature_dims = {
        'vitals': train_dataset.vitals_dim,
        'lab': train_dataset.lab_dim,
        'drug': train_dataset.drug_dim,
        'text': 768,  # 假设使用BERT类模型的文本嵌入维度
        'kg': 64      # 默认知识图谱嵌入维度
    }
    
    return data_loaders, feature_dims 
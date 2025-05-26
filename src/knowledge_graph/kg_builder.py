#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识图谱构建模块
用于构建和处理医学知识图谱
"""

import os
import random
import pickle
import numpy as np
import torch
import logging

# 创建日志记录器
logger = logging.getLogger(__name__)

class MedicalKnowledgeGraph:
    """
    医学知识图谱类
    用于表示和处理医学领域的知识图谱
    """
    def __init__(self):
        """初始化知识图谱"""
        self.entity2id = {}  # 实体到ID的映射
        self.id2entity = {}  # ID到实体的映射
        self.relation2id = {}  # 关系到ID的映射
        self.id2relation = {}  # ID到关系的映射
        self.entity_types = {}  # 实体类型
        self.triples = []  # 三元组列表
        
    def add_entity(self, entity_name, entity_type=None):
        """
        添加实体
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
        """
        if entity_name not in self.entity2id:
            entity_id = len(self.entity2id)
            self.entity2id[entity_name] = entity_id
            self.id2entity[entity_id] = entity_name
            if entity_type:
                self.entity_types[entity_name] = entity_type
            return entity_id
        return self.entity2id[entity_name]
        
    def add_relation(self, relation_name):
        """
        添加关系
        
        Args:
            relation_name: 关系名称
        """
        if relation_name not in self.relation2id:
            relation_id = len(self.relation2id)
            self.relation2id[relation_name] = relation_id
            self.id2relation[relation_id] = relation_name
            return relation_id
        return self.relation2id[relation_name]
        
    def add_triple(self, head, relation, tail):
        """
        添加三元组
        
        Args:
            head: 头实体名称
            relation: 关系名称
            tail: 尾实体名称
        """
        # 确保实体和关系存在
        head_id = self.add_entity(head)
        tail_id = self.add_entity(tail)
        relation_id = self.add_relation(relation)
        
        # 添加三元组
        triple = (head, relation, tail)
        if triple not in self.triples:
            self.triples.append(triple)
            
    def save(self, output_file):
        """
        保存知识图谱
        
        Args:
            output_file: 输出文件路径
        """
        data = {
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'relation2id': self.relation2id,
            'id2relation': self.id2relation,
            'entity_types': self.entity_types,
            'triples': self.triples
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"知识图谱已保存到 {output_file}")
        
    @classmethod
    def load(cls, input_file):
        """
        加载知识图谱
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            kg: 知识图谱对象
        """
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        kg.entity2id = data['entity2id']
        kg.id2entity = data['id2entity']
        kg.relation2id = data['relation2id']
        kg.id2relation = data['id2relation']
        kg.entity_types = data['entity_types']
        kg.triples = data['triples']
        
        logger.info(f"从 {input_file} 加载了知识图谱，包含{len(kg.entity2id)}个实体，{len(kg.triples)}个关系")
        
        return kg

def build_medical_ontology():
    """
    构建医学本体
    包含疾病、症状、药物等基本医学概念
    
    Returns:
        kg: 医学本体知识图谱
    """
    kg = MedicalKnowledgeGraph()
    
    # 添加脓毒症相关实体和关系
    # 疾病
    kg.add_entity("Sepsis", "Disease")
    kg.add_entity("SepticShock", "Disease")
    
    # 症状
    kg.add_entity("Fever", "Symptom")
    kg.add_entity("Tachycardia", "Symptom")
    kg.add_entity("Tachypnea", "Symptom")
    kg.add_entity("Hypotension", "Symptom")
    kg.add_entity("Hyperlactatemia", "Symptom")
    kg.add_entity("AcuteKidneyInjury", "Symptom")
    kg.add_entity("Thrombocytopenia", "Symptom")
    kg.add_entity("Leukocytosis", "Symptom")
    kg.add_entity("Leukopenia", "Symptom")
    kg.add_entity("Hyperbilirubinemia", "Symptom")
    
    # 药物
    # 抗生素
    kg.add_entity("Ceftriaxone", "Antibiotic")
    kg.add_entity("Vancomycin", "Antibiotic")
    kg.add_entity("Piperacillin", "Antibiotic")
    kg.add_entity("Meropenem", "Antibiotic")
    kg.add_entity("Ciprofloxacin", "Antibiotic")
    kg.add_entity("Levofloxacin", "Antibiotic")
    
    # 升压药
    kg.add_entity("Norepinephrine", "Vasopressor")
    kg.add_entity("Epinephrine", "Vasopressor")
    kg.add_entity("Dopamine", "Vasopressor")
    kg.add_entity("Vasopressin", "Vasopressor")
    
    # 其他
    kg.add_entity("Lactate", "LabValue")
    kg.add_entity("Platelet", "LabValue")
    kg.add_entity("WBC", "LabValue")
    kg.add_entity("Creatinine", "LabValue")
    kg.add_entity("BUN", "LabValue")
    kg.add_entity("Bilirubin", "LabValue")
    
    # 添加关系
    # 疾病-症状关系
    for symptom in ["Fever", "Tachycardia", "Tachypnea", "Hypotension", 
                   "Hyperlactatemia", "AcuteKidneyInjury", "Thrombocytopenia",
                   "Leukocytosis", "Leukopenia", "Hyperbilirubinemia"]:
        kg.add_triple("Sepsis", "has_symptom", symptom)
    
    # 脓毒性休克是脓毒症的一种特殊类型
    kg.add_triple("SepticShock", "is_a", "Sepsis")
    kg.add_triple("SepticShock", "has_symptom", "Hypotension")
    
    # 疾病-治疗关系
    # 抗生素治疗
    for antibiotic in ["Ceftriaxone", "Vancomycin", "Piperacillin", 
                      "Meropenem", "Ciprofloxacin", "Levofloxacin"]:
        kg.add_triple("Sepsis", "treated_by", antibiotic)
    
    # 升压药治疗（主要用于脓毒性休克）
    for vasopressor in ["Norepinephrine", "Epinephrine", "Dopamine", "Vasopressin"]:
        kg.add_triple("SepticShock", "treated_by", vasopressor)
    
    # 药物类型关系
    for antibiotic in ["Ceftriaxone", "Vancomycin", "Piperacillin", 
                      "Meropenem", "Ciprofloxacin", "Levofloxacin"]:
        kg.add_triple(antibiotic, "is_a", "Antibiotic")
    
    for vasopressor in ["Norepinephrine", "Epinephrine", "Dopamine", "Vasopressin"]:
        kg.add_triple(vasopressor, "is_a", "Vasopressor")
    
    # 实验室检查关系
    kg.add_triple("Lactate", "indicates", "Hyperlactatemia")
    kg.add_triple("Platelet", "indicates", "Thrombocytopenia")
    kg.add_triple("WBC", "indicates", "Leukocytosis")
    kg.add_triple("WBC", "indicates", "Leukopenia")
    kg.add_triple("Creatinine", "indicates", "AcuteKidneyInjury")
    kg.add_triple("BUN", "indicates", "AcuteKidneyInjury")
    kg.add_triple("Bilirubin", "indicates", "Hyperbilirubinemia")
    
    logger.info(f"医学本体构建完成，包含{len(kg.entity2id)}个实体，{len(kg.triples)}个关系")
    
    return kg

def enrich_with_patient_data(kg, dataset):
    """
    使用患者数据丰富知识图谱
    
    Args:
        kg: 知识图谱对象
        dataset: 患者数据集
        
    Returns:
        enriched_kg: 丰富后的知识图谱
    """
    logger.info("使用患者数据丰富知识图谱...")
    
    # 已添加的关系，避免重复
    added_relations = set()
    
    # 生命体征阈值
    vital_thresholds = {
        'heart_rate': 100,  # 心动过速阈值 >100
        'resp_rate': 22,    # 呼吸急促阈值 >22
        'temperature': 38,  # 发热阈值 >38°C
        'sbp': 90,          # 低收缩压阈值 <90
        'dbp': 60           # 低舒张压阈值 <60
    }
    
    # 实验室值阈值
    lab_thresholds = {
        'wbc': {'low': 4, 'high': 12},  # 白细胞计数，低值<4，高值>12
        'creatinine': 1.5,  # 肌酐 >1.5
        'bun': 25,          # 尿素氮 >25
        'lactate': 2.0,     # 乳酸 >2.0
        'platelet': 100,    # 血小板 <100
        'bilirubin': 1.5    # 胆红素 >1.5
    }
    
    # 生命体征到实体的映射
    vital_to_entity = {
        'heart_rate': 'Tachycardia',
        'resp_rate': 'Tachypnea',
        'temperature': 'Fever',
        'sbp': 'Hypotension',
        'dbp': 'Hypotension'
    }
    
    # 实验室值到实体的映射
    lab_to_entity = {
        'wbc': ['Leukocytosis', 'Leukopenia'],  # 白细胞增多/减少
        'creatinine': 'AcuteKidneyInjury',
        'bun': 'AcuteKidneyInjury',
        'lactate': 'Hyperlactatemia',
        'platelet': 'Thrombocytopenia',
        'bilirubin': 'Hyperbilirubinemia'
    }
    
    # 药物到实体的映射
    drug_to_entity = {
        'antibiotic': ['Ceftriaxone', 'Vancomycin', 'Piperacillin'],
        'vasopressor': ['Norepinephrine', 'Epinephrine', 'Dopamine']
    }
    
    # 检查并适配不同类型的数据集
    has_vitals_features = hasattr(dataset, 'vitals_features')
    has_lab_features = hasattr(dataset, 'lab_features')
    has_drug_features = hasattr(dataset, 'drug_features')
    
    # 如果没有特征列表，则使用默认值
    default_vitals_features = ['heart_rate', 'resp_rate', 'temperature', 'sbp', 'dbp', 'spo2']
    default_lab_features = ['wbc', 'creatinine', 'bun', 'lactate', 'platelet', 'bilirubin'] 
    default_drug_features = ['antibiotic', 'vasopressor']
    
    # 处理每个患者的数据
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            
            # 创建患者实体ID（用索引作为ID）
            patient_id = f"Patient_{i}"
            kg.add_entity(patient_id, "Patient")
            
            # 如果有脓毒症，添加关系
            if torch.any(sample['labels'] > 0):
                kg.add_triple(patient_id, "diagnosed_with", "Sepsis")
            
            # 处理生命体征
            vitals = sample['vitals'].cpu().numpy()
            vitals_features = dataset.vitals_features if has_vitals_features else default_vitals_features
            
            for feature_idx, feature_name in enumerate(vitals_features):
                if feature_idx >= vitals.shape[1]:
                    # 跳过超出范围的特征
                    continue
                    
                if feature_name in vital_to_entity and feature_name in vital_thresholds:
                    # 取所有时间点的均值
                    feature_vals = vitals[:, feature_idx]
                    feature_vals = feature_vals[feature_vals != 0]  # 排除零值
                    
                    if len(feature_vals) > 0:
                        mean_val = np.mean(feature_vals)
                        threshold = vital_thresholds[feature_name]
                        
                        # 根据不同生命体征的判断条件
                        if feature_name in ['sbp', 'dbp']:
                            # 低血压判断
                            if mean_val < threshold:
                                entity_name = vital_to_entity[feature_name]
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
                        else:
                            # 高值判断（心动过速、呼吸急促、发热）
                            if mean_val > threshold:
                                entity_name = vital_to_entity[feature_name]
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
            
            # 处理实验室值
            labs = sample['labs'].cpu().numpy()
            lab_features = dataset.lab_features if has_lab_features else default_lab_features
            
            for feature_idx, feature_name in enumerate(lab_features):
                if feature_idx >= labs.shape[1]:
                    # 跳过超出范围的特征
                    continue
                    
                if feature_name in lab_to_entity:
                    # 取所有时间点的均值
                    feature_vals = labs[:, feature_idx]
                    feature_vals = feature_vals[feature_vals != 0]  # 排除零值
                    
                    if len(feature_vals) > 0:
                        mean_val = np.mean(feature_vals)
                        
                        # 根据是单向阈值还是双向阈值进行判断
                        if feature_name == 'wbc':
                            thresholds = lab_thresholds[feature_name]
                            if mean_val > thresholds['high']:
                                entity_name = lab_to_entity[feature_name][0]  # Leukocytosis
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
                            elif mean_val < thresholds['low']:
                                entity_name = lab_to_entity[feature_name][1]  # Leukopenia
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
                        elif feature_name == 'platelet':
                            if mean_val < lab_thresholds[feature_name]:
                                entity_name = lab_to_entity[feature_name]
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
                        else:
                            if mean_val > lab_thresholds[feature_name]:
                                entity_name = lab_to_entity[feature_name]
                                relation_key = (patient_id, "has_symptom", entity_name)
                                if relation_key not in added_relations:
                                    kg.add_triple(patient_id, "has_symptom", entity_name)
                                    added_relations.add(relation_key)
            
            # 处理药物使用
            drugs = sample['drugs'].cpu().numpy()
            drug_features = dataset.drug_features if has_drug_features else default_drug_features
            
            for feature_idx, feature_name in enumerate(drug_features):
                if feature_idx >= drugs.shape[1]:
                    # 跳过超出范围的特征
                    continue
                    
                if feature_name in drug_to_entity:
                    # 如果在任何时间点使用了该类药物
                    if np.any(drugs[:, feature_idx] > 0):
                        # 随机选择一种该类药物（简化处理）
                        entity_names = drug_to_entity[feature_name]
                        entity_name = np.random.choice(entity_names)
                        relation_key = (patient_id, "receives", entity_name)
                        if relation_key not in added_relations:
                            kg.add_triple(patient_id, "receives", entity_name)
                            added_relations.add(relation_key)
        except Exception as e:
            logger.warning(f"处理患者 {i} 数据时出错: {e}，跳过该患者")
            continue
    
    # 由于可能的错误处理，暂时取消添加患者相似性
    # add_patient_similarities(kg, dataset)
    
    logger.info(f"知识图谱丰富完成，现包含{len(kg.entity2id)}个实体，{len(kg.triples)}个关系")
    return kg 

def add_patient_similarities(kg, dataset, similarity_threshold=0.7):
    """
    添加患者之间的相似性关系
    基于症状和实验室值计算患者相似度
    
    Args:
        kg: 知识图谱对象
        dataset: 患者数据集
        similarity_threshold: 相似度阈值，超过该值才添加关系
    """
    logger.info("计算患者相似性并添加到知识图谱...")
    
    # 已添加的相似性关系，避免重复
    added_similarities = set()
    
    # 只处理一部分患者，避免组合数过多
    max_patients = min(1000, len(dataset))
    
    # 检查数据集中是否包含必要的键
    try:
        sample = dataset[0]
    except Exception as e:
        logger.warning(f"无法获取样本: {e}，取消添加患者相似性")
        return
    
    # 确保必要的数据存在
    try:
        for i in range(max_patients):
            try:
                # 获取患者i的数据
                data_i = dataset[i]
                vitals_i = data_i['vitals'].cpu().numpy()
                labs_i = data_i['labs'].cpu().numpy()
                labels_i = data_i['labels'].cpu().numpy()
                
                # 计算患者i的特征向量：使用非零值的均值
                vitals_i_mean = []
                for j in range(vitals_i.shape[1]):
                    vals = vitals_i[:, j]
                    vals = vals[vals != 0]
                    vitals_i_mean.append(np.mean(vals) if len(vals) > 0 else 0)
                
                labs_i_mean = []
                for j in range(labs_i.shape[1]):
                    vals = labs_i[:, j]
                    vals = vals[vals != 0]
                    labs_i_mean.append(np.mean(vals) if len(vals) > 0 else 0)
                
                # 组合特征
                features_i = np.array(vitals_i_mean + labs_i_mean, dtype=np.float32)
                
                # 患者i的脓毒症标签
                has_sepsis_i = np.any(labels_i > 0)
                
                # 与其他患者比较
                for j in range(i+1, max_patients):
                    try:
                        # 获取患者j的数据
                        data_j = dataset[j]
                        vitals_j = data_j['vitals'].cpu().numpy()
                        labs_j = data_j['labs'].cpu().numpy()
                        labels_j = data_j['labels'].cpu().numpy()
                        
                        # 计算患者j的特征向量
                        vitals_j_mean = []
                        for k in range(vitals_j.shape[1]):
                            vals = vitals_j[:, k]
                            vals = vals[vals != 0]
                            vitals_j_mean.append(np.mean(vals) if len(vals) > 0 else 0)
                        
                        labs_j_mean = []
                        for k in range(labs_j.shape[1]):
                            vals = labs_j[:, k]
                            vals = vals[vals != 0]
                            labs_j_mean.append(np.mean(vals) if len(vals) > 0 else 0)
                        
                        # 组合特征
                        features_j = np.array(vitals_j_mean + labs_j_mean, dtype=np.float32)
                        
                        # 患者j的脓毒症标签
                        has_sepsis_j = np.any(labels_j > 0)
                        
                        # 计算相似度
                        if len(features_i) > 0 and len(features_j) > 0:
                            # 处理向量长度不同的情况
                            min_len = min(len(features_i), len(features_j))
                            features_i_norm = features_i[:min_len]
                            features_j_norm = features_j[:min_len]
                            
                            # 避免零向量
                            if np.all(features_i_norm == 0) or np.all(features_j_norm == 0):
                                continue
                                
                            # 计算余弦相似度
                            dot_product = np.dot(features_i_norm, features_j_norm)
                            norm_i = np.linalg.norm(features_i_norm)
                            norm_j = np.linalg.norm(features_j_norm)
                            
                            if norm_i > 0 and norm_j > 0:
                                similarity = dot_product / (norm_i * norm_j)
                                
                                # 如果相似度超过阈值且两者脓毒症状态相同，添加相似关系
                                if similarity > similarity_threshold and has_sepsis_i == has_sepsis_j:
                                    patient_i_id = f"Patient_{i}"
                                    patient_j_id = f"Patient_{j}"
                                    relation_key = (patient_i_id, "similar_to", patient_j_id)
                                    
                                    if relation_key not in added_similarities:
                                        kg.add_triple(patient_i_id, "similar_to", patient_j_id)
                                        added_similarities.add(relation_key)
                    except Exception as e:
                        logger.warning(f"比较患者 {i} 和 {j} 时出错: {e}，跳过该比较")
                        continue
            except Exception as e:
                logger.warning(f"处理患者 {i} 特征时出错: {e}，跳过该患者")
                continue
        
        logger.info(f"添加了{len(added_similarities)}个患者相似性关系")
    
    except Exception as e:
        logger.warning(f"添加患者相似性时发生错误: {e}")
        # 如果出错，直接返回原始知识图谱
        return 

def build_knowledge_graph(dataset, ontology_only=False, output_dir="./output"):
    """
    构建医学知识图谱
    
    Args:
        dataset: 患者数据集
        ontology_only: 是否只构建本体，不包含患者数据
        output_dir: 输出目录
        
    Returns:
        kg: 知识图谱对象
    """
    logger.info("构建医学本体...")
    kg = build_medical_ontology()
    logger.info(f"医学本体构建完成，包含{len(kg.entity2id)}个实体，{len(kg.triples)}个关系")
    
    if not ontology_only:
        logger.info("使用患者数据丰富知识图谱...")
        try:
            kg = enrich_with_patient_data(kg, dataset)
            
            # 尝试添加患者相似性
            try:
                add_patient_similarities(kg, dataset)
            except Exception as e:
                logger.warning(f"添加患者相似性时出错: {e}")
        except Exception as e:
            logger.error(f"使用患者数据丰富知识图谱时出错: {e}")
    
    # 保存知识图谱
    os.makedirs(output_dir, exist_ok=True)
    kg_path = os.path.join(output_dir, "knowledge_graph.pkl")
    kg.save(kg_path)
    logger.info(f"知识图谱已保存到 {kg_path}")
    
    return kg 
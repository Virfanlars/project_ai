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
from .external_knowledge import ExternalMedicalKnowledgeBase
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

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

class KnowledgeGraphBuilder:
    """
    知识图谱构建器类，整合外部医学知识库
    """
    
    def __init__(self, umls_api_key: Optional[str] = None):
        """
        初始化知识图谱构建器
        
        Args:
            umls_api_key: UMLS API密钥
        """
        self.graph = nx.DiGraph()
        self.external_kb = ExternalMedicalKnowledgeBase(umls_api_key)
        self.feature_mappings = {}
        self.concept_cache = {}
        
    def build_sepsis_knowledge_graph(self) -> nx.DiGraph:
        """
        构建脓毒症相关的医学知识图谱
        
        Returns:
            构建好的知识图谱
        """
        logger.info("开始构建脓毒症知识图谱...")
        
        try:
            # 获取脓毒症相关概念
            sepsis_concepts = self.external_kb.get_sepsis_related_concepts()
            
            # 添加概念节点
            self._add_concept_nodes(sepsis_concepts)
            
            # 构建概念间关系
            self._build_concept_relations(sepsis_concepts)
            
            # 添加特征映射
            self._build_feature_mappings()
            
            logger.info(f"知识图谱构建完成，包含 {self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            raise Exception(f"无法构建知识图谱: {e}")
    
    def _add_concept_nodes(self, concepts_dict: Dict[str, List[Dict[str, Any]]]):
        """
        添加概念节点到图中
        
        Args:
            concepts_dict: 按来源分类的概念字典
        """
        for source, concepts in concepts_dict.items():
            for concept in concepts:
                # 生成节点ID
                node_id = self._generate_node_id(concept, source)
                
                # 提取概念名称
                name = self._extract_concept_name(concept, source)
                
                # 添加节点属性
                node_attrs = {
                    'name': name,
                    'source': source,
                    'type': 'medical_concept',
                    'original_data': concept
                }
                
                # 添加特定于来源的属性
                if source == 'umls':
                    node_attrs['cui'] = concept.get('cui', '')
                elif source == 'snomed':
                    node_attrs['sctid'] = concept.get('sctid', '')
                    node_attrs['fsn'] = concept.get('fsn', '')
                elif source == 'rxnorm_drugs':
                    node_attrs['rxcui'] = concept.get('rxcui', '')
                    node_attrs['tty'] = concept.get('tty', '')
                elif 'clinical_' in source:
                    node_attrs['clinical_id'] = concept.get('id', '')
                
                self.graph.add_node(node_id, **node_attrs)
                
                # 缓存概念
                self.concept_cache[name.lower()] = node_id
    
    def _generate_node_id(self, concept: Dict[str, Any], source: str) -> str:
        """
        生成节点ID
        
        Args:
            concept: 概念数据
            source: 数据来源
            
        Returns:
            节点ID
        """
        if source == 'umls':
            return f"umls_{concept.get('cui', '')}"
        elif source == 'snomed':
            return f"snomed_{concept.get('sctid', '')}"
        elif source == 'rxnorm_drugs':
            return f"rxnorm_{concept.get('rxcui', '')}"
        else:
            # Clinical Tables等其他来源
            concept_id = concept.get('id', '')
            return f"{source}_{concept_id}"
    
    def _extract_concept_name(self, concept: Dict[str, Any], source: str) -> str:
        """
        提取概念名称
        
        Args:
            concept: 概念数据
            source: 数据来源
            
        Returns:
            概念名称
        """
        if source == 'umls':
            return concept.get('name', '')
        elif source == 'snomed':
            return concept.get('name', '') or concept.get('fsn', '')
        elif source == 'rxnorm_drugs':
            return concept.get('name', '')
        else:
            return concept.get('name', '') or concept.get('id', '')
    
    def _build_concept_relations(self, concepts_dict: Dict[str, List[Dict[str, Any]]]):
        """
        构建概念间的关系
        
        Args:
            concepts_dict: 概念字典
        """
        logger.info("构建概念间关系...")
        
        # 为每个概念查询关系
        for source, concepts in concepts_dict.items():
            for concept in concepts[:5]:  # 限制数量以避免过多API调用
                try:
                    concept_id = self._get_concept_id_for_relations(concept, source)
                    if concept_id:
                        relations = self.external_kb.get_concept_relations(concept_id, source)
                        self._add_relations_to_graph(concept, source, relations)
                        
                except Exception as e:
                    logger.warning(f"获取概念关系失败 {concept}: {e}")
                    continue
        
        # 添加基于名称的语义关系
        self._add_semantic_relations()
    
    def _get_concept_id_for_relations(self, concept: Dict[str, Any], source: str) -> Optional[str]:
        """
        获取用于查询关系的概念ID
        
        Args:
            concept: 概念数据
            source: 数据来源
            
        Returns:
            概念ID
        """
        if source == 'umls':
            return concept.get('cui')
        elif source == 'snomed':
            return concept.get('sctid')
        return None
    
    def _add_relations_to_graph(self, source_concept: Dict[str, Any], source: str, relations: List[Dict[str, Any]]):
        """
        将关系添加到图中
        
        Args:
            source_concept: 源概念
            source: 数据来源
            relations: 关系列表
        """
        source_node_id = self._generate_node_id(source_concept, source)
        
        for relation in relations:
            try:
                # 提取目标概念信息
                if source == 'umls':
                    target_cui = relation.get('target_cui', '')
                    if target_cui:
                        target_node_id = f"umls_{target_cui}"
                        relation_type = relation.get('relation_type', 'related_to')
                elif source == 'snomed':
                    target_sctid = relation.get('target_sctid', '')
                    if target_sctid:
                        target_node_id = f"snomed_{target_sctid}"
                        relation_type = relation.get('relation_type', 'related_to')
                else:
                    continue
                
                # 如果目标节点不存在，创建一个基本节点
                if target_node_id not in self.graph:
                    self.graph.add_node(target_node_id, 
                                      name=f"concept_{target_node_id}", 
                                      source=source,
                                      type='external_concept')
                
                # 添加边
                self.graph.add_edge(source_node_id, target_node_id, 
                                  relation_type=relation_type,
                                  source=source)
                
            except Exception as e:
                logger.warning(f"添加关系失败: {e}")
                continue
    
    def _add_semantic_relations(self):
        """
        基于概念名称添加语义关系
        """
        logger.info("添加语义关系...")
        
        # 定义语义关系规则
        semantic_rules = {
            'sepsis': ['infection', 'inflammation', 'shock'],
            'infection': ['antibiotic', 'fever', 'leukocytosis'],
            'shock': ['hypotension', 'vasopressor'],
            'organ failure': ['kidney', 'liver', 'lung'],
            'antibiotic': ['treatment', 'antimicrobial'],
        }
        
        # 应用语义规则
        for concept1, related_terms in semantic_rules.items():
            concept1_nodes = self._find_nodes_by_name_pattern(concept1)
            
            for term in related_terms:
                concept2_nodes = self._find_nodes_by_name_pattern(term)
                
                for node1 in concept1_nodes:
                    for node2 in concept2_nodes:
                        if node1 != node2:
                            self.graph.add_edge(node1, node2, 
                                              relation_type='semantic_related',
                                              source='inferred')
    
    def _find_nodes_by_name_pattern(self, pattern: str) -> List[str]:
        """
        根据名称模式查找节点
        
        Args:
            pattern: 搜索模式
            
        Returns:
            匹配的节点ID列表
        """
        matching_nodes = []
        pattern_lower = pattern.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            if pattern_lower in node_name:
                matching_nodes.append(node_id)
        
        return matching_nodes
    
    def _build_feature_mappings(self):
        """
        构建数据特征到医学概念的映射
        """
        logger.info("构建特征映射...")
        
        # 定义特征到概念的映射规则
        feature_mappings = {
            # 生命体征
            'heart_rate': ['tachycardia', 'bradycardia', 'cardiac'],
            'blood_pressure': ['hypotension', 'hypertension'],
            'temperature': ['fever', 'hypothermia'],
            'respiratory_rate': ['tachypnea', 'respiratory'],
            'oxygen_saturation': ['hypoxia', 'respiratory failure'],
            
            # 实验室检查
            'white_blood_cell': ['leukocytosis', 'leukopenia', 'infection'],
            'lactate': ['hyperlactatemia', 'tissue hypoxia'],
            'creatinine': ['kidney dysfunction', 'renal failure'],
            'bilirubin': ['liver dysfunction', 'hepatic failure'],
            'platelet': ['thrombocytopenia', 'coagulopathy'],
            'procalcitonin': ['bacterial infection', 'sepsis'],
            
            # 治疗
            'antibiotic': ['antimicrobial therapy', 'infection treatment'],
            'vasopressor': ['shock treatment', 'hypotension'],
            'fluid': ['resuscitation', 'volume expansion'],
            'mechanical_ventilation': ['respiratory support', 'ARDS'],
        }
        
        # 为每个特征查找对应的概念节点
        for feature, concept_terms in feature_mappings.items():
            related_nodes = []
            
            for term in concept_terms:
                nodes = self._find_nodes_by_name_pattern(term)
                related_nodes.extend(nodes)
            
            if related_nodes:
                self.feature_mappings[feature] = list(set(related_nodes))
        
        logger.info(f"构建了 {len(self.feature_mappings)} 个特征映射")
    
    def get_concept_embeddings_for_features(self, features: List[str]) -> Dict[str, np.ndarray]:
        """
        获取特征对应的概念嵌入
        
        Args:
            features: 特征名称列表
            
        Returns:
            特征到嵌入的映射
        """
        embeddings = {}
        
        for feature in features:
            if feature in self.feature_mappings:
                # 获取相关概念节点
                concept_nodes = self.feature_mappings[feature]
                
                # 计算概念嵌入（这里用简单的节点度数作为示例）
                concept_embedding = np.zeros(128)  # 假设128维嵌入
                
                for node in concept_nodes:
                    if node in self.graph:
                        # 使用节点的连接信息生成简单嵌入
                        node_degree = self.graph.degree(node)
                        node_centrality = nx.degree_centrality(self.graph).get(node, 0)
                        
                        # 生成基于图结构的特征向量
                        node_features = np.array([
                            node_degree,
                            node_centrality,
                            len(list(self.graph.neighbors(node))),
                            # 添加更多图特征...
                        ])
                        
                        # 扩展到128维（重复和变换）
                        if len(node_features) < 128:
                            repeat_times = 128 // len(node_features)
                            remainder = 128 % len(node_features)
                            concept_embedding += np.concatenate([
                                np.tile(node_features, repeat_times),
                                node_features[:remainder]
                            ])
                
                # 归一化
                if len(concept_nodes) > 0:
                    concept_embedding /= len(concept_nodes)
                    concept_embedding = concept_embedding / (np.linalg.norm(concept_embedding) + 1e-8)
                
                embeddings[feature] = concept_embedding
        
        return embeddings
    
    def save_graph(self, filepath: str):
        """
        保存知识图谱
        
        Args:
            filepath: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存图和映射
            data_to_save = {
                'graph': self.graph,
                'feature_mappings': self.feature_mappings,
                'concept_cache': self.concept_cache
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
                
            logger.info(f"知识图谱已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            raise
    
    def load_graph(self, filepath: str) -> bool:
        """
        加载知识图谱
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"知识图谱文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = data['graph']
            self.feature_mappings = data['feature_mappings']
            self.concept_cache = data['concept_cache']
            
            logger.info(f"知识图谱已从 {filepath} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            return False
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            统计信息字典
        """
        if self.graph.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0}
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'feature_mappings': len(self.feature_mappings),
            'sources': {}
        }
        
        # 按来源统计节点
        for node, data in self.graph.nodes(data=True):
            source = data.get('source', 'unknown')
            if source not in stats['sources']:
                stats['sources'][source] = 0
            stats['sources'][source] += 1
        
        return stats
    
    def find_concept_by_name(self, name: str) -> Optional[str]:
        """
        根据名称查找概念节点
        
        Args:
            name: 概念名称
            
        Returns:
            节点ID或None
        """
        name_lower = name.lower()
        
        # 首先查找缓存
        if name_lower in self.concept_cache:
            return self.concept_cache[name_lower]
        
        # 在图中搜索
        for node_id, node_data in self.graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            if name_lower in node_name or node_name in name_lower:
                self.concept_cache[name_lower] = node_id
                return node_id
        
        return None
    
    def get_related_concepts(self, concept_name: str, max_depth: int = 2) -> List[Tuple[str, str, int]]:
        """
        获取相关概念
        
        Args:
            concept_name: 概念名称
            max_depth: 最大搜索深度
            
        Returns:
            相关概念列表 (节点ID, 概念名称, 距离)
        """
        start_node = self.find_concept_by_name(concept_name)
        if not start_node:
            return []
        
        related_concepts = []
        
        try:
            # 使用广度优先搜索查找相关概念
            for target_node in self.graph.nodes():
                if target_node != start_node:
                    try:
                        path_length = nx.shortest_path_length(self.graph, start_node, target_node)
                        if path_length <= max_depth:
                            node_data = self.graph.nodes[target_node]
                            concept_name = node_data.get('name', target_node)
                            related_concepts.append((target_node, concept_name, path_length))
                    except nx.NetworkXNoPath:
                        continue
        except Exception as e:
            logger.warning(f"搜索相关概念时出错: {e}")
        
        # 按距离排序
        related_concepts.sort(key=lambda x: x[2])
        return related_concepts

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
    
    logger.info(f"构建医学本体完成，包含{len(kg.entity2id)}个实体")
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
        'sbp': 'Hypotension'
    }
    
    # 处理生命体征数据
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
                    
                try:
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
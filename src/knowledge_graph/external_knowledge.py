"""
外部医学知识库集成
集成多个外部医学数据库和本体库
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
import json
from urllib.parse import urlencode, quote
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class ExternalMedicalKnowledgeBase:
    """集成多个外部医学知识库的统一接口"""
    
    def __init__(self, umls_api_key: Optional[str] = None):
        """
        初始化外部医学知识库
        
        Args:
            umls_api_key: UMLS UTS API密钥
        """
        self.umls_api_key = umls_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SepsisRiskPrediction/1.0 (Medical Research)',
            'Accept': 'application/json'
        })
        
        # API端點配置
        self.endpoints = {
            'umls_uts': 'https://uts-ws.nlm.nih.gov/rest',
            'clinical_tables': 'https://clinicaltables.nlm.nih.gov/api',
            'snomed_browser': 'https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser',
            'rxnorm': 'https://rxnav.nlm.nih.gov/REST'
        }
        
        # 緩存
        self._cache = {}
        
        # 初始化本地緩存數據庫
        self._init_local_cache()
    
    def _init_local_cache(self):
        """初始化本地緩存數據庫"""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(exist_ok=True)
        self.cache_db_path = cache_dir / "medical_knowledge_cache.db"
        
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_cache (
                    query TEXT PRIMARY KEY,
                    source TEXT,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relation_cache (
                    source_cui TEXT,
                    target_cui TEXT,
                    relation_type TEXT,
                    source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_cui, target_cui, relation_type, source)
                )
            """)
    
    def get_umls_ticket(self) -> Optional[str]:
        """獲取UMLS UTS認證票據"""
        if not self.umls_api_key:
            logger.warning("未提供UMLS API密鑰")
            return None
        
        try:
            auth_url = f"{self.endpoints['umls_uts']}/authentication"
            response = self.session.post(
                auth_url,
                data={'apikey': self.umls_api_key}
            )
            
            if response.status_code == 201:
                ticket_url = response.headers.get('location')
                if ticket_url:
                    ticket = ticket_url.split('/')[-1]
                    logger.info("成功獲取UMLS認證票據")
                    return ticket
            
            logger.error(f"獲取UMLS票據失敗: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"UMLS認證錯誤: {e}")
            return None
    
    def search_umls_concepts(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        在UMLS中搜索醫學概念
        
        Args:
            term: 搜索詞
            limit: 結果數量限制
            
        Returns:
            概念列表
        """
        cache_key = f"umls_search_{term}_{limit}"
        cached = self._get_cached_result(cache_key, 'umls')
        if cached:
            return cached
        
        ticket = self.get_umls_ticket()
        if not ticket:
            return []
        
        try:
            search_url = f"{self.endpoints['umls_uts']}/search/current"
            params = {
                'string': term,
                'ticket': ticket,
                'pageSize': limit,
                'includeObsolete': 'false',
                'includeSuppressible': 'false'
            }
            
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if 'result' in data and 'results' in data['result']:
                    for result in data['result']['results']:
                        concept = {
                            'cui': result.get('ui', ''),
                            'name': result.get('name', ''),
                            'source': 'UMLS',
                            'semantic_types': []
                        }
                        results.append(concept)
                
                self._cache_result(cache_key, 'umls', results)
                logger.info(f"從UMLS找到 {len(results)} 個概念")
                return results
            
            logger.error(f"UMLS搜索失敗: {response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"UMLS搜索錯誤: {e}")
            return []
    
    def search_clinical_tables(self, table_name: str, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        在NLM Clinical Tables中搜索
        
        Args:
            table_name: 表名 (如 'conditions', 'procedures', 'rxterms')
            term: 搜索詞
            limit: 結果數量限制
            
        Returns:
            結果列表
        """
        cache_key = f"clinical_tables_{table_name}_{term}_{limit}"
        cached = self._get_cached_result(cache_key, 'clinical_tables')
        if cached:
            return cached
        
        try:
            search_url = f"{self.endpoints['clinical_tables']}/{table_name}/v3/search"
            params = {
                'terms': term,
                'maxList': limit
            }
            
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if isinstance(data, list) and len(data) > 3:
                    # Clinical Tables API返回格式: [total_count, records, headers, ...]
                    records = data[1]
                    headers = data[2] if len(data) > 2 else []
                    
                    for record in records:
                        if isinstance(record, list) and len(record) > 0:
                            result = {
                                'id': record[0] if len(record) > 0 else '',
                                'name': record[1] if len(record) > 1 else record[0],
                                'source': f'ClinicalTables_{table_name}',
                                'data': dict(zip(headers, record)) if headers else {}
                            }
                            results.append(result)
                
                self._cache_result(cache_key, 'clinical_tables', results)
                logger.info(f"從Clinical Tables找到 {len(results)} 個結果")
                return results
            
            logger.error(f"Clinical Tables搜索失敗: {response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"Clinical Tables搜索錯誤: {e}")
            return []
    
    def search_snomed_concepts(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        在SNOMED CT中搜索概念
        注意：由於SNOMED CT國際版需要許可，我們使用Clinical Tables中的HPO作為替代
        
        Args:
            term: 搜索詞
            limit: 結果數量限制
            
        Returns:
            概念列表
        """
        cache_key = f"snomed_search_{term}_{limit}"
        cached = self._get_cached_result(cache_key, 'snomed')
        if cached:
            return cached
        
        try:
            # 使用Clinical Tables的HPO (Human Phenotype Ontology) 作為SNOMED CT的替代
            # 因為SNOMED CT國際版需要許可證
            search_url = f"{self.endpoints['clinical_tables']}/hpo/v3/search"
            params = {
                'terms': term,
                'maxList': limit
            }
            
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if isinstance(data, list) and len(data) > 1:
                    # Clinical Tables API返回格式: [total_count, records, headers, ...]
                    records = data[1]
                    
                    for record in records:
                        if isinstance(record, list) and len(record) >= 2:
                            concept = {
                                'hpo_id': record[0],
                                'name': record[1],
                                'source': 'HPO_ClinicalTables',
                                'type': 'phenotype'
                            }
                            results.append(concept)
                
                self._cache_result(cache_key, 'snomed', results)
                logger.info(f"從HPO找到 {len(results)} 個表型概念")
                return results
            
            logger.warning(f"HPO搜索失敗: {response.status_code}，將嘗試使用疾病名稱數據庫")
            
            # 如果HPO失敗，嘗試疾病名稱數據庫
            disease_url = f"{self.endpoints['clinical_tables']}/disease_names/v3/search"
            response = self.session.get(disease_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if isinstance(data, list) and len(data) > 1:
                    records = data[1]
                    for record in records:
                        if isinstance(record, list) and len(record) >= 2:
                            concept = {
                                'disease_id': record[1] if len(record) > 1 else '',
                                'name': record[0],
                                'source': 'DiseaseNames_ClinicalTables',
                                'type': 'disease'
                            }
                            results.append(concept)
                
                self._cache_result(cache_key, 'snomed', results)
                logger.info(f"從疾病名稱數據庫找到 {len(results)} 個概念")
                return results
            
            logger.error("所有醫學概念搜索方法都失敗")
            return []
            
        except Exception as e:
            logger.error(f"醫學概念搜索錯誤: {e}")
            # 如果所有方法都失敗，返回空列表但不崩潰
            logger.warning("由於SNOMED CT需要許可證且替代方法也失敗，返回空結果")
            return []
    
    def search_rxnorm_drugs(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        在RxNorm中搜索藥物
        
        Args:
            term: 搜索詞
            limit: 結果數量限制
            
        Returns:
            藥物列表
        """
        cache_key = f"rxnorm_search_{term}_{limit}"
        cached = self._get_cached_result(cache_key, 'rxnorm')
        if cached:
            return cached
        
        try:
            search_url = f"{self.endpoints['rxnorm']}/drugs.json"
            params = {
                'name': term,
                'maxEntries': limit
            }
            
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if 'drugGroup' in data and 'conceptGroup' in data['drugGroup']:
                    for group in data['drugGroup']['conceptGroup']:
                        if 'conceptProperties' in group:
                            for prop in group['conceptProperties']:
                                drug = {
                                    'rxcui': prop.get('rxcui', ''),
                                    'name': prop.get('name', ''),
                                    'synonym': prop.get('synonym', ''),
                                    'tty': prop.get('tty', ''),
                                    'source': 'RxNorm'
                                }
                                results.append(drug)
                
                self._cache_result(cache_key, 'rxnorm', results)
                logger.info(f"從RxNorm找到 {len(results)} 個藥物")
                return results
            
            logger.error(f"RxNorm搜索失敗: {response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"RxNorm搜索錯誤: {e}")
            return []
    
    def get_sepsis_related_concepts(self) -> Dict[str, List[Dict[str, Any]]]:
        """獲取與脓毒症相關的醫學概念"""
        sepsis_terms = [
            'sepsis', 'septic shock', 'bacteremia', 'endotoxemia',
            'systemic inflammatory response syndrome', 'SIRS',
            'organ dysfunction', 'multiple organ failure',
            'acute kidney injury', 'acute respiratory failure',
            'hypotension', 'tachycardia', 'fever', 'hypothermia'
        ]
        
        all_concepts = {
            'umls': [],
            'snomed': [],
            'clinical_conditions': [],
            'rxnorm_drugs': []
        }
        
        for term in sepsis_terms:
            logger.info(f"搜索脓毒症相關概念: {term}")
            
            # 搜索UMLS
            umls_results = self.search_umls_concepts(term, limit=5)
            all_concepts['umls'].extend(umls_results)
            
            # 搜索SNOMED CT
            snomed_results = self.search_snomed_concepts(term, limit=5)
            all_concepts['snomed'].extend(snomed_results)
            
            # 搜索Clinical Tables conditions
            conditions = self.search_clinical_tables('conditions', term, limit=5)
            all_concepts['clinical_conditions'].extend(conditions)
            
            # 搜索相關藥物
            if term in ['sepsis', 'septic shock', 'bacteremia']:
                drugs = self.search_rxnorm_drugs('antibiotic', limit=3)
                all_concepts['rxnorm_drugs'].extend(drugs)
            
            # 避免過於頻繁的API調用
            time.sleep(0.5)
        
        # 去重
        for category in all_concepts:
            seen = set()
            unique_concepts = []
            for concept in all_concepts[category]:
                key = concept.get('cui') or concept.get('sctid') or concept.get('rxcui') or concept.get('id', '')
                if key and key not in seen:
                    seen.add(key)
                    unique_concepts.append(concept)
            all_concepts[category] = unique_concepts
        
        logger.info(f"獲取到脓毒症相關概念總數: {sum(len(v) for v in all_concepts.values())}")
        return all_concepts
    
    def get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        獲取藥物相互作用信息
        
        Args:
            drug_name: 藥物名稱
            
        Returns:
            相互作用列表
        """
        try:
            # 首先在RxNorm中查找藥物
            drugs = self.search_rxnorm_drugs(drug_name, limit=1)
            if not drugs:
                return []
            
            rxcui = drugs[0].get('rxcui')
            if not rxcui:
                return []
            
            # 查詢相互作用
            interaction_url = f"{self.endpoints['rxnorm']}/interaction/interaction.json"
            params = {'rxcui': rxcui}
            
            response = self.session.get(interaction_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                interactions = []
                
                if 'interactionTypeGroup' in data:
                    for group in data['interactionTypeGroup']:
                        if 'interactionType' in group:
                            for interaction_type in group['interactionType']:
                                if 'interactionPair' in interaction_type:
                                    for pair in interaction_type['interactionPair']:
                                        interaction = {
                                            'drug1': pair.get('interactionConcept', [{}])[0].get('minConceptItem', {}).get('name', ''),
                                            'drug2': pair.get('interactionConcept', [{}])[1].get('minConceptItem', {}).get('name', '') if len(pair.get('interactionConcept', [])) > 1 else '',
                                            'severity': pair.get('severity', ''),
                                            'description': pair.get('description', ''),
                                            'source': 'RxNorm'
                                        }
                                        interactions.append(interaction)
                
                logger.info(f"找到 {len(interactions)} 個藥物相互作用")
                return interactions
            
            return []
            
        except Exception as e:
            logger.error(f"獲取藥物相互作用錯誤: {e}")
            return []
    
    def _get_cached_result(self, query: str, source: str) -> Optional[List[Dict[str, Any]]]:
        """從緩存獲取結果"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    "SELECT result FROM concept_cache WHERE query = ? AND source = ? AND datetime(timestamp) > datetime('now', '-1 day')",
                    (query, source)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.error(f"緩存讀取錯誤: {e}")
        return None
    
    def _cache_result(self, query: str, source: str, result: List[Dict[str, Any]]):
        """緩存結果"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO concept_cache (query, source, result) VALUES (?, ?, ?)",
                    (query, source, json.dumps(result))
                )
        except Exception as e:
            logger.error(f"緩存保存錯誤: {e}")
    
    def get_concept_relations(self, concept_id: str, source: str = 'umls') -> List[Dict[str, Any]]:
        """
        獲取概念之間的關係
        
        Args:
            concept_id: 概念ID
            source: 數據源
            
        Returns:
            關係列表
        """
        if source == 'umls':
            return self._get_umls_relations(concept_id)
        elif source == 'snomed':
            return self._get_snomed_relations(concept_id)
        else:
            return []
    
    def _get_umls_relations(self, cui: str) -> List[Dict[str, Any]]:
        """獲取UMLS概念關係"""
        ticket = self.get_umls_ticket()
        if not ticket:
            return []
        
        try:
            relations_url = f"{self.endpoints['umls_uts']}/content/current/CUI/{cui}/relations"
            params = {'ticket': ticket}
            
            response = self.session.get(relations_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                relations = []
                
                if 'result' in data:
                    for relation in data['result']:
                        rel = {
                            'source_cui': cui,
                            'target_cui': relation.get('relatedId', ''),
                            'relation_type': relation.get('relationLabel', ''),
                            'source': 'UMLS'
                        }
                        relations.append(rel)
                
                return relations
            
            return []
            
        except Exception as e:
            logger.error(f"獲取UMLS關係錯誤: {e}")
            return []
    
    def _get_snomed_relations(self, sctid: str) -> List[Dict[str, Any]]:
        """獲取SNOMED CT概念關係"""
        try:
            relations_url = f"{self.endpoints['snomed_browser']}/MAIN/concepts/{sctid}/relationships"
            params = {'activeFilter': 'true'}
            
            response = self.session.get(relations_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                relations = []
                
                if 'items' in data:
                    for relation in data['items']:
                        rel = {
                            'source_sctid': sctid,
                            'target_sctid': relation.get('target', {}).get('conceptId', ''),
                            'relation_type': relation.get('type', {}).get('pt', {}).get('term', ''),
                            'source': 'SNOMED_CT'
                        }
                        relations.append(rel)
                
                return relations
            
            return []
            
        except Exception as e:
            logger.error(f"獲取SNOMED CT關係錯誤: {e}")
            return []

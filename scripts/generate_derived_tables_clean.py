import os
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库配置 - 针对MIMIC-IV真实数据
DATABASE_CONFIG = {
    # 真实MIMIC-IV数据库连接信息
    'host': 'localhost',
    'port': 5432,
    'database': 'mimic',
    'user': 'postgres',
    'password': 'postgres',
    
    # 表模式映射
    'schema_map': {
        'hosp': 'mimiciv_hosp',      # 医院数据
        'icu': 'mimiciv_icu',        # ICU数据
        'derived': 'mimiciv_derived' # 派生表
    },
    
    # 性能和规模设置
    'min_patient_count': 10000,      # 增加患者数量
    'sample_size': None,             # 不限制样本大小，使用全部数据
    'use_fast_query': True,          # 使用优化的查询
    'use_materialized_views': True,  # 使用物化视图
    'load_all_tables': True,         # 加载所有相关表
    'use_parallel_processing': True, # 启用并行处理
    'chunk_size': 50000,             # 增加数据块大小，减少IO操作
    
    # 重要：禁用模拟数据，使用真实数据
    'use_mock_data': False,          # 不使用模拟数据
    'local_mode': False              # 不使用本地模式
}

# 覆盖环境变量设置
os.environ['SEPSIS_USE_MOCK_DATA'] = 'false'
os.environ['SEPSIS_LOCAL_MODE'] = 'false'

# 从环境变量获取数据库连接配置（如果有）
if os.environ.get('MIMIC_DB_HOST'):
    DATABASE_CONFIG['host'] = os.environ.get('MIMIC_DB_HOST')
if os.environ.get('MIMIC_DB_PORT'):
    DATABASE_CONFIG['port'] = int(os.environ.get('MIMIC_DB_PORT'))
if os.environ.get('MIMIC_DB_NAME'):
    DATABASE_CONFIG['database'] = os.environ.get('MIMIC_DB_NAME')
if os.environ.get('MIMIC_DB_USER'):
    DATABASE_CONFIG['user'] = os.environ.get('MIMIC_DB_USER')
if os.environ.get('MIMIC_DB_PASSWORD'):
    DATABASE_CONFIG['password'] = os.environ.get('MIMIC_DB_PASSWORD')

# 生成连接字符串
def get_connection_string():
    """
    生成数据库连接字符串
    
    返回:
        str: PostgreSQL连接字符串
    """
    config = DATABASE_CONFIG
    conn_string = f"host={config['host']} port={config['port']} dbname={config['database']} user={config['user']} password={config['password']}"
    return conn_string

# 获取数据库连接
def get_connection():
    """
    创建到MIMIC-IV数据库的连接
    
    如果无法连接到真实数据库，会回退到模拟连接并发出警告
    
    返回:
        connection: 数据库连接对象
    """
    try:
        import psycopg2
        conn_string = get_connection_string()
        logger.info(f"正在连接MIMIC-IV数据库: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
        
        # 尝试连接
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
        )
        
        # 测试连接
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result and result[0] == 1:
            logger.info("MIMIC-IV数据库连接成功!")
            cursor.close()
            return conn
        cursor.close()
        
    except ImportError as e:
        logger.error(f"未安装psycopg2模块，无法连接PostgreSQL: {e}")
        logger.error("请安装: pip install psycopg2 或 pip install psycopg2-binary")
        
    except Exception as e:
        logger.error(f"MIMIC-IV数据库连接失败: {e}")
        
    # 如果明确要求使用真实数据但连接失败
    if os.environ.get('SEPSIS_REQUIRE_REAL_DATA', '').lower() == 'true':
        logger.error("必须使用真实数据，但无法连接到MIMIC-IV数据库。终止程序。")
        sys.exit(1)
    
    logger.warning("无法连接到真实数据库，将回退到模拟数据库连接")
    logger.warning("⚠️ 警告: 使用模拟数据将影响结果质量和可靠性")
    return get_mock_connection()

# 模拟数据库连接（仅在真实连接不可用时使用）
def get_mock_connection():
    """
    创建模拟数据库连接，用于开发和测试
    
    返回:
        MockConnection: 模拟连接对象
    """
    logger.info("创建MIMIC-IV模拟数据库连接")
    
    class MockCursor:
        """模拟数据库游标"""
        def __init__(self):
            self.result = None
            
        def execute(self, query, params=None):
            """执行模拟查询"""
            query_lower = query.lower()
            if "information_schema.schemata" in query_lower:
                # 模拟检查模式是否存在
                self.result = [(True,)]
            elif "information_schema.tables" in query_lower:
                # 模拟检查表是否存在
                self.result = [(True,)]
            elif "count(*)" in query_lower:
                # 模拟计数查询
                if "sepsis3" in query_lower:
                    self.result = [(5000,)]  # 假设5000例脓毒症病例
                elif "patients" in query_lower:
                    self.result = [(70000,)]  # 假设70000名患者
                else:
                    self.result = [(10000,)]  # 其他表默认10000行
            elif "mimic_version" in query_lower:
                # 模拟获取MIMIC版本
                self.result = [("MIMIC-IV v2.2",)]
            elif "select 1" in query_lower:
                # 测试连接
                self.result = [(1,)]
            else:
                # 默认空结果
                self.result = []
            
        def fetchall(self):
            """获取所有结果"""
            return self.result if hasattr(self, 'result') and self.result is not None else []
            
        def fetchone(self):
            """获取一行结果"""
            return self.result[0] if hasattr(self, 'result') and self.result and len(self.result) > 0 else None
            
        def close(self):
            """关闭游标"""
            pass
    
    class MockConnection:
        """模拟数据库连接"""
        def cursor(self):
            """获取游标"""
            return MockCursor()
            
        def close(self):
            """关闭连接"""
            pass
            
        def commit(self):
            """提交事务"""
            pass
            
        def rollback(self):
            """回滚事务"""
            pass
    
    return MockConnection()

# 数据库查询优化设置
def get_query_optimizations():
    """
    返回PostgreSQL查询优化设置
    
    返回:
        dict: 优化参数字典
    """
    return {
        'enable_nestloop': 'off',                 # 禁用嵌套循环连接
        'enable_seqscan': 'off',                  # 禁用顺序扫描，优先使用索引
        'random_page_cost': '1.0',                # 降低随机页访问成本
        'effective_cache_size': '8GB',            # 增加缓存大小估计
        'work_mem': '2GB',                        # 增加工作内存
        'maintenance_work_mem': '1GB',            # 增加维护工作内存
        'max_parallel_workers_per_gather': '8',   # 增加并行工作线程
        'max_parallel_workers': '16',             # 增加最大并行工作线程
        'max_parallel_maintenance_workers': '8'   # 增加维护并行工作线程
    }
# 创建一个名为create_config.py的文件
with open('utils/database_config.py', 'w', encoding='utf-8') as f:
    f.write('''# 数据库配置文件
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DATABASE_CONFIG = {
    'host': '172.16.3.67',
    'port': 5432,
    'database': 'mimiciv',
    'user': 'postgres',
    'password': 'mimic',
    'schema_map': {
        'hosp': 'mimiciv_hosp',
        'icu': 'mimiciv_icu',
        'derived': 'mimiciv_derived'
    },
    'min_patient_count': 10000,
    'sample_size': None,
    'use_fast_query': True,
    'use_materialized_views': True,
    'load_all_tables': True,
    'use_parallel_processing': True,
    'chunk_size': 50000
}

def get_connection_string():
    """生成数据库连接字符串"""
    config = DATABASE_CONFIG
    host = os.environ.get('MIMIC_DB_HOST', config['host'])
    port = os.environ.get('MIMIC_DB_PORT', config['port'])
    dbname = os.environ.get('MIMIC_DB_NAME', config['database'])
    user = os.environ.get('MIMIC_DB_USER', config['user'])
    password = os.environ.get('MIMIC_DB_PASSWORD', config['password'])
    conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    return conn_string

def get_connection():
    """创建数据库连接，失败时直接退出程序"""
    try:
        import psycopg2
        conn_string = get_connection_string()
        logger.info(f"尝试连接数据库: {conn_string}")
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
        )
        logger.info("数据库连接成功")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        logger.error("系统需要有效的数据库连接才能运行，程序退出")
        sys.exit(1)

def get_query_optimizations():
    """返回查询优化设置"""
    return {
        'enable_nestloop': 'off',
        'enable_seqscan': 'off',
        'random_page_cost': '1.0',
        'effective_cache_size': '8GB',
        'work_mem': '2GB',
        'maintenance_work_mem': '1GB',
        'max_parallel_workers_per_gather': '8',
        'max_parallel_workers': '16',
        'max_parallel_maintenance_workers': '8'
    }
''')
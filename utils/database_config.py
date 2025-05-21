import os
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration for MIMIC-IV
DATABASE_CONFIG = {
    # Real MIMIC-IV database connection info
    'host': 'localhost',  # Database host
    'port': 5432,
    'database': 'mimiciv',
    'user': 'postgres',
    'password': '123456',
    
    # Schema mapping
    'schema_map': {
        'hosp': 'mimiciv_hosp',      # Hospital data
        'icu': 'mimiciv_icu',        # ICU data
        'derived': 'mimiciv_derived' # Derived tables
    },
    
    # Performance and scale settings
    'min_patient_count': 10000,      # Increase patient count
    'sample_size': None,             # No limit on sample size, use all data
    'use_fast_query': True,          # Use optimized queries
    'use_materialized_views': True,  # Use materialized views
    'load_all_tables': True,         # Load all related tables
    'use_parallel_processing': True, # Enable parallel processing
    'chunk_size': 50000,             # Increase data chunk size, reduce IO operations
    
    # Important: Disable simulated data, use real data
    'use_mock_data': False,          # Do not use mock data
    'local_mode': False              # Do not use local mode
}

# Override with environment variables
os.environ['SEPSIS_USE_MOCK_DATA'] = 'false'
os.environ['SEPSIS_LOCAL_MODE'] = 'false'

# Get database connection config from environment variables (if available)
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

# Generate connection string
def get_connection_string():
    """
    Generate database connection string
    
    Returns:
        str: PostgreSQL connection string
    """
    config = DATABASE_CONFIG
    conn_string = f"host={config['host']} port={config['port']} dbname={config['database']} user={config['user']} password={config['password']}"
    return conn_string

# Get database connection
def get_connection():
    """
    Create connection to MIMIC-IV database
    
    If unable to connect to real database, will fall back to mock connection and issue warning
    
    Returns:
        connection: Database connection object
    """
    try:
        import psycopg2
        conn_string = get_connection_string()
        logger.info(f"Connecting to MIMIC-IV database: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
        
        # Try to connect
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            database=DATABASE_CONFIG['database'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password']
        )
        
        # Test connection
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result and result[0] == 1:
            logger.info("MIMIC-IV database connection successful!")
            cursor.close()
            return conn
        cursor.close()
        
    except ImportError as e:
        logger.error(f"psycopg2 module not installed, cannot connect to PostgreSQL: {e}")
        logger.error("Please install: pip install psycopg2 or pip install psycopg2-binary")
        
    except Exception as e:
        logger.error(f"MIMIC-IV database connection failed: {e}")
        
    # If real data is explicitly required but connection failed
    if os.environ.get('SEPSIS_REQUIRE_REAL_DATA', '').lower() == 'true' or os.environ.get('SEPSIS_REQUIRE_REAL_DATA') == '1':
        logger.error("Real data is required but cannot connect to MIMIC-IV database. Terminating program.")
        sys.exit(1)
    
    logger.warning("Unable to connect to real database, falling back to mock database connection")
    logger.warning("WARNING: Using mock data will affect the quality and reliability of results")
    return get_mock_connection()

# Mock database connection (only used when real connection is unavailable)
def get_mock_connection():
    """
    Create mock database connection, for development and testing
    
    Returns:
        MockConnection: Mock connection object
    """
    logger.info("Creating MIMIC-IV mock database connection")
    
    class MockCursor:
        """Mock database cursor"""
        def __init__(self):
            self.result = None
            
        def execute(self, query, params=None):
            """Execute mock query"""
            query_lower = query.lower()
            if "information_schema.schemata" in query_lower:
                # Mock checking if schema exists
                self.result = [(True,)]
            elif "information_schema.tables" in query_lower:
                # Mock checking if table exists
                self.result = [(True,)]
            elif "count(*)" in query_lower:
                # Mock count queries
                if "sepsis3" in query_lower:
                    self.result = [(5000,)]  # Assume 5000 sepsis cases
                elif "patients" in query_lower:
                    self.result = [(70000,)]  # Assume 70000 patients
                else:
                    self.result = [(10000,)]  # Default 10000 rows for other tables
            elif "mimic_version" in query_lower:
                # Mock getting MIMIC version
                self.result = [("MIMIC-IV v2.2",)]
            elif "select 1" in query_lower:
                # Test connection
                self.result = [(1,)]
            else:
                # Default empty result
                self.result = []
            
        def fetchall(self):
            """Get all results"""
            return self.result if hasattr(self, 'result') and self.result is not None else []
            
        def fetchone(self):
            """Get one row of results"""
            return self.result[0] if hasattr(self, 'result') and self.result and len(self.result) > 0 else None
            
        def close(self):
            """Close cursor"""
            pass
    
    class MockConnection:
        """Mock database connection"""
        def cursor(self):
            """Get cursor"""
            return MockCursor()
            
        def close(self):
            """Close connection"""
            pass
            
        def commit(self):
            """Commit transaction"""
            pass
            
        def rollback(self):
            """Rollback transaction"""
            pass
    
    return MockConnection()

# Database query optimization settings
def get_query_optimizations():
    """
    Return PostgreSQL query optimization settings
    
    Returns:
        dict: Optimization parameter dictionary
    """
    return {
        'enable_nestloop': 'off',                 # Disable nested loop joins
        'enable_seqscan': 'off',                  # Disable sequential scans, prioritize indexes
        'random_page_cost': '1.0',                # Lower random page access cost
        'effective_cache_size': '8GB',            # Increase cache size estimate
        'work_mem': '1999MB',                     # Working memory (changed from 2GB to stay within PostgreSQL limits)
        'maintenance_work_mem': '1GB',            # Increase maintenance work memory
        'max_parallel_workers_per_gather': '8',   # Increase parallel worker threads
        'max_parallel_workers': '16',             # Increase maximum parallel worker threads
        'max_parallel_maintenance_workers': '8'   # Increase maintenance parallel worker threads
    }
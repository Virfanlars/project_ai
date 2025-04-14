#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sepsis Table Structure Inspector
This script connects to the MIMIC-IV database and inspects the structure 
of the sepsis3 table to determine correct field names.
"""

import sys
import os
import psycopg2
import logging
from prettytable import PrettyTable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import database configuration
try:
    from utils.database_config import get_connection, DATABASE_CONFIG
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def inspect_table_structure():
    """
    Inspect the structure of the sepsis3 table
    """
    conn = None
    try:
        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get schema name
        derived_schema = DATABASE_CONFIG['schema_map']['derived']
        
        # Get table columns
        cursor.execute(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_schema = '{derived_schema}' 
        AND table_name = 'sepsis3'
        ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        if not columns:
            logger.error(f"Table {derived_schema}.sepsis3 not found or no columns returned")
            return
            
        # Display table structure
        logger.info(f"Structure of {derived_schema}.sepsis3 table:")
        table = PrettyTable()
        table.field_names = ["Column Name", "Data Type", "Nullable"]
        
        for column in columns:
            table.add_row(column)
            
        print(table)
        
        # Sample data
        logger.info("Fetching sample data (first 5 rows):")
        cursor.execute(f"""
        SELECT * FROM {derived_schema}.sepsis3
        LIMIT 5
        """)
        
        rows = cursor.fetchall()
        if rows:
            # Get column names
            col_names = [desc[0] for desc in cursor.description]
            
            # Display sample data
            sample_table = PrettyTable()
            sample_table.field_names = col_names
            
            for row in rows:
                sample_table.add_row(row)
                
            print(sample_table)
            
        # Check for time-related columns
        time_columns = [col[0] for col in columns if 'time' in col[0].lower()]
        if time_columns:
            logger.info(f"Time-related columns found: {', '.join(time_columns)}")
        else:
            logger.warning("No time-related columns found in sepsis3 table")
            
    except Exception as e:
        logger.error(f"Error inspecting table structure: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    inspect_table_structure() 
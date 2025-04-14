#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库连接测试工具
此脚本使用硬编码的连接参数，不依赖可能有编码问题的配置文件
"""

import os
import sys
import psycopg2
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def test_connection():
    """测试与MIMIC-IV数据库的连接"""
    print("MIMIC-IV数据库连接测试")
    print("=" * 50)
    
    # 硬编码连接参数，避免使用可能有编码问题的配置文件
    host = "172.16.3.67"
    port = 5432
    dbname = "mimiciv"
    user = "postgres"
    password = "123456"
    
    print(f"尝试连接到 {host}:{port}/{dbname} 数据库...")
    
    try:
        # 构建连接字符串
        conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        print(f"使用连接字符串: {conn_string}")
        
        # 尝试连接
        start_time = time.time()
        conn = psycopg2.connect(conn_string)
        end_time = time.time()
        
        print(f"连接成功！耗时: {end_time - start_time:.2f}秒")
        
        # 获取数据库信息
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"数据库版本: {version}")
        
        # 检查schema是否存在
        print("\n检查数据库schema...")
        cursor.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name LIKE 'mimic%';
        """)
        schemas = cursor.fetchall()
        
        if schemas:
            print("找到以下MIMIC相关schema:")
            for schema in schemas:
                print(f"  - {schema[0]}")
        else:
            print("警告: 未找到任何MIMIC相关schema")
        
        # 检查mimiciv_derived是否存在
        print("\n检查mimiciv_derived schema...")
        cursor.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name = 'mimiciv_derived';
        """)
        derived_schema = cursor.fetchone()
        
        if derived_schema:
            print("mimiciv_derived schema存在")
            
            # 检查sepsis3表是否存在
            cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'mimiciv_derived' 
            AND table_name = 'sepsis3';
            """)
            sepsis3_table = cursor.fetchone()
            
            if sepsis3_table:
                print("sepsis3表存在")
                
                # 获取表中的行数
                cursor.execute("SELECT COUNT(*) FROM mimiciv_derived.sepsis3;")
                row_count = cursor.fetchone()[0]
                print(f"sepsis3表中有 {row_count} 行数据")
                
                # 获取sepsis3表中sepsis3=1的行数
                cursor.execute("SELECT COUNT(*) FROM mimiciv_derived.sepsis3 WHERE sepsis3 = 1;")
                sepsis_row_count = cursor.fetchone()[0]
                print(f"sepsis3表中有 {sepsis_row_count} 行脓毒症患者数据 (sepsis3 = 1)")
                
                # 获取表结构
                cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'mimiciv_derived' 
                AND table_name = 'sepsis3';
                """)
                columns = cursor.fetchall()
                
                print("\nsepsis3表结构:")
                for col in columns:
                    print(f"  - {col[0]} ({col[1]})")
            else:
                print("警告: sepsis3表不存在!")
        else:
            print("警告: mimiciv_derived schema不存在!")
        
        # 关闭连接
        conn.close()
        print("\n测试完成！数据库连接正常工作")
        
    except Exception as e:
        print(f"连接失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection() 
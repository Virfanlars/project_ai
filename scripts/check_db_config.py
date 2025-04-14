#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库配置检查工具
此脚本检查数据库配置并打印信息，帮助诊断连接问题
"""

import os
import sys
import traceback

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def main():
    print("数据库配置检查工具")
    print("=" * 50)
    
    # 手动构建连接字符串，避免导入可能出现编码问题的模块
    host = "172.16.3.67"
    port = 5432
    dbname = "mimiciv"
    user = "postgres"
    password = "123456"  # 使用配置中提供的密码
    
    print(f"数据库配置:")
    print(f"  主机: {host}")
    print(f"  端口: {port}")
    print(f"  数据库名: {dbname}")
    print(f"  用户名: {user}")
    print(f"  密码: {'*' * len(password)}")
    
    # 构建标准连接字符串
    conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    print(f"\n生成的连接字符串 (不含特殊字符):")
    print(conn_string)
    
    # 检查 utils/database_config.py 文件
    db_config_path = os.path.join(project_root, "utils", "database_config.py")
    if os.path.exists(db_config_path):
        print(f"\n检测到数据库配置文件: {db_config_path}")
        
        try:
            # 尝试二进制模式读取，检查是否存在非UTF-8字符
            with open(db_config_path, 'rb') as f:
                content = f.read()
                
            try:
                # 尝试以UTF-8解码
                decoded = content.decode('utf-8')
                print("文件可以用UTF-8解码，没有发现编码问题")
            except UnicodeDecodeError as e:
                print(f"文件编码出现问题: {e}")
                print("文件中包含非UTF-8字符，需要修复")
                
                # 显示问题位置周围的内容
                problem_pos = e.start
                context_range = 20  # 显示错误周围的字节
                start = max(0, problem_pos - context_range)
                end = min(len(content), problem_pos + context_range)
                
                print(f"\n问题字节位置: {problem_pos}")
                print(f"问题字节: 0x{content[problem_pos]:02x}")
                print(f"问题字节周围内容 (十六进制):")
                hex_dump = ' '.join([f"{b:02x}" for b in content[start:end]])
                print(hex_dump)
                
                print("\n建议: 打开文件，全选内容，复制，然后删除全部内容，粘贴回去，并以UTF-8格式保存")
        except Exception as e:
            print(f"分析文件时出错: {e}")
            traceback.print_exc()
    else:
        print(f"\n警告: 未找到数据库配置文件 {db_config_path}")
    
    print("\n推荐操作:")
    print("1. 运行下面的命令修复数据库配置文件的编码:")
    print(f"   python -c \"with open('{db_config_path}', 'r', errors='ignore') as f, open('{db_config_path}.new', 'w', encoding='utf-8') as o: o.write(f.read()); import os; os.replace('{db_config_path}.new', '{db_config_path}')\"")
    print("2. 在脚本中手动设置数据库连接参数:")
    print("   host='172.16.3.67', port=5432, dbname='mimiciv', user='postgres', password='123456'")
    print("3. 使用简单的连接设置运行测试脚本:")
    print(f"   python scripts/test_db_connection.py")

if __name__ == "__main__":
    main() 
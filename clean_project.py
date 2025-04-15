#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目清理脚本
移除项目中未使用的文件，保留核心功能所需的文件
"""

import os
import shutil
import sys

# 保留的核心文件和目录
ESSENTIAL_FILES = [
    # 根目录核心文件
    "README.md",
    "requirements.txt",
    "config.py",
    "run_complete_sepsis_system.py",
    "set_mimic_env.ps1",
    "set_mimic_env_fixed.bat",
    "clean_project.py",
    "__init__.py",
    
    # scripts目录
    "scripts/fixed_extract_sepsis_data.py",
    "scripts/process_sepsis_data.py",
    "scripts/convert_processed_data.py",
    "scripts/test_db_connection.py",
    "scripts/check_db_config.py",
    "scripts/model_training.py",
    "scripts/generate_derived_tables.py",
    "scripts/__init__.py",
    
    # utils目录
    "utils/database_config.py",
    "utils/data_loading.py",
    "utils/dataset.py",
    "utils/evaluation.py",
    "utils/explanation.py",
    "utils/visualization.py",
    "utils/__init__.py",
]

# 保留的目录
ESSENTIAL_DIRS = [
    "data",
    "data/processed",
    "data/knowledge_graph",
    "models",
    "results",
    "results/figures",
    "logs",
    "scripts",
    "utils"
]

def normalize_path(path):
    """标准化路径，确保使用正斜杠"""
    return path.replace('\\', '/')

def should_keep_file(file_path):
    """判断是否应该保留文件"""
    normalized_path = normalize_path(file_path)
    
    # 检查是否在必要文件列表中
    for essential_file in ESSENTIAL_FILES:
        essential_normalized = normalize_path(essential_file)
        if normalized_path == essential_normalized or normalized_path.endswith(essential_normalized):
            return True
            
    # 排除某些隐藏目录和临时文件
    if "/__pycache__/" in normalized_path or "/.git/" in normalized_path:
        return True
        
    return False

def backup_project():
    """备份项目"""
    print("创建项目备份...")
    
    # 创建备份目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    backup_dir = os.path.join(project_root, "project_backup")
    
    # 如果备份目录已存在，先删除
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    # 创建新的备份
    shutil.copytree(project_root, backup_dir, ignore=shutil.ignore_patterns('project_backup', '.git', '__pycache__'))
    
    print(f"项目已备份到: {backup_dir}")
    return backup_dir

def create_essential_dirs():
    """创建必要的目录"""
    print("确保必要目录存在...")
    
    for dir_path in ESSENTIAL_DIRS:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

def remove_unnecessary_files():
    """移除不必要的文件"""
    print("移除不必要的文件...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    removed_count = 0
    
    # 遍历所有文件
    for root, dirs, files in os.walk(project_root, topdown=True):
        # 排除备份目录和.git目录
        if "project_backup" in root or ".git" in root:
            continue
            
        # 检查并移除文件
        for file in files:
            file_path = os.path.join(root, file)
            if not should_keep_file(file_path):
                try:
                    os.remove(file_path)
                    print(f"已移除: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"无法移除 {file_path}: {e}")
    
    print(f"共移除了 {removed_count} 个不必要的文件")

def main():
    """主函数"""
    print("==== 项目清理工具 ====")
    print("警告: 此脚本将移除项目中不必要的文件。")
    response = input("是否继续? (y/n): ")
    
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 备份项目
    backup_dir = backup_project()
    
    # 确保必要目录存在
    create_essential_dirs()
    
    # 移除不必要的文件
    remove_unnecessary_files()
    
    print("\n清理完成!")
    print(f"如需恢复原始项目，请从备份目录恢复: {backup_dir}")

if __name__ == "__main__":
    main() 
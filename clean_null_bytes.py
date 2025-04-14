#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理Python文件中的null字节
"""

import os
import sys

def clean_file(file_path):
    """清理文件中的null字节"""
    try:
        print(f"清理文件: {file_path}")
        
        # 尝试以二进制模式读取文件
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # 移除null字节
        cleaned_content = content.replace(b'\x00', b'')
        
        if content != cleaned_content:
            # 内容已更改，写回文件
            with open(file_path, 'wb') as f:
                f.write(cleaned_content)
            print(f"✓ 已清理文件: {file_path}")
            return True
        else:
            print(f"✓ 文件无需清理: {file_path}")
            return False
    except Exception as e:
        print(f"✗ 清理文件时出错: {file_path}，错误: {e}")
        return False

def clean_directory(directory, extensions=['.py']):
    """清理目录中所有指定扩展名的文件"""
    cleaned_files = 0
    failed_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if clean_file(file_path):
                    cleaned_files += 1
                else:
                    failed_files += 1
    
    return cleaned_files, failed_files

if __name__ == "__main__":
    print("===== Python文件Null字节清理工具 =====")
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            print(f"清理目录: {target}")
            cleaned, failed = clean_directory(target)
            print(f"\n清理完成! 处理了 {cleaned + failed} 个文件，成功清理 {cleaned} 个文件")
        elif os.path.isfile(target):
            print(f"清理文件: {target}")
            if clean_file(target):
                print("\n清理成功!")
            else:
                print("\n清理失败或文件无需清理")
        else:
            print(f"错误: {target} 不是有效的文件或目录")
    else:
        # 默认清理当前目录的所有Python文件
        current_dir = os.getcwd()
        print(f"未指定目标，清理当前目录: {current_dir}")
        cleaned, failed = clean_directory(current_dir)
        print(f"\n清理完成! 处理了 {cleaned + failed} 个文件，成功清理 {cleaned} 个文件") 
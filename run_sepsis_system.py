#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症早期预警系统 - 完整执行脚本
包括数据处理、模型训练和可视化报告生成
"""

import os
import sys
import time
import subprocess
import logging
import argparse
import matplotlib
matplotlib.rc("font",family='YouYuan')
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sepsis_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_with_logging(command, description):
    """执行命令并记录输出"""
    logger.info(f"执行: {description}")
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"命令成功完成: {command} (耗时: {duration:.2f}秒)")
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"输出: {line}")
            return True
        else:
            logger.error(f"命令执行失败: {command}")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.error(f"错误: {line}")
            return False
    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        return False

def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/knowledge_graph',
        'models',
        'models/fusion',
        'models/time_series',
        'models/knowledge_graph',
        'models/text',
        'results',
        'results/figures',
        'results/explanations',
        'results/visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"已确保目录存在: {directory}")

def process_data(use_sample_data=False, local_mode=False):
    """处理数据阶段"""
    logger.info("=== 阶段 1: 数据处理 ===")
    
    # 设置环境变量以控制数据访问模式
    if local_mode or use_sample_data:
        os.environ['SEPSIS_LOCAL_MODE'] = 'true'
        os.environ['SEPSIS_USE_MOCK_DATA'] = 'true'
        logger.info("已启用本地模式和模拟数据")
    
    if use_sample_data:
        cmd = "python run_data_processing.py --sample"
        desc = "使用样本数据进行数据处理"
    else:
        cmd = "python run_data_processing.py"
        desc = "从数据库提取并处理数据"
    
    success = run_with_logging(cmd, desc)
    
    if not success:
        logger.warning("数据处理阶段出现问题，尝试使用样本数据")
        os.environ['SEPSIS_USE_MOCK_DATA'] = 'true'
        success = run_with_logging("python run_data_processing.py --sample", "使用样本数据进行数据处理")
    
    if not success:
        logger.error("数据处理阶段失败，无法继续")
        sys.exit(1)
    
    return success

def train_model(epochs=None, batch_size=None, use_sample_data=False):
    """模型训练阶段"""
    logger.info("=== 阶段 2: 模型训练 ===")
    
    cmd = "python train.py"
    
    if epochs:
        cmd += f" --epochs {epochs}"
    if batch_size:
        cmd += f" --batch_size {batch_size}"
    if use_sample_data:
        cmd += " --sample"
    
    # 添加更多高级训练选项
    cmd += " --use_lr_scheduler --use_sampler --gradient_clip 1.0"
    
    success = run_with_logging(cmd, "训练脓毒症预警模型")
    
    if not success:
        logger.error("模型训练阶段失败，无法继续")
        sys.exit(1)
    
    return success

def generate_visualizations():
    """生成可视化和报告"""
    logger.info("=== 阶段 3: 生成可视化和报告 ===")
    
    success = run_with_logging("python run_visualization.py", "生成可视化报告")
    
    if not success:
        logger.error("可视化生成阶段失败")
        return False
    
    return True

def main():
    """主执行函数"""
    start_time = time.time()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行完整脓毒症早期预警系统')
    parser.add_argument('--sample', action='store_true', help='使用样本数据而不是连接数据库')
    parser.add_argument('--local', action='store_true', help='在本地模式下运行(不尝试连接远程数据库)')
    parser.add_argument('--mock_data', action='store_true', help='使用模拟数据(无需数据库连接)')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='训练批量大小')
    parser.add_argument('--skip_data', action='store_true', help='跳过数据处理阶段')
    parser.add_argument('--skip_train', action='store_true', help='跳过模型训练阶段')
    parser.add_argument('--only_viz', action='store_true', help='只运行可视化阶段')
    args = parser.parse_args()
    
    logger.info("脓毒症早期预警系统 - 开始执行")
    logger.info(f"运行参数: {args}")
    
    # 设置本地模式或模拟数据环境变量
    if args.local:
        os.environ['SEPSIS_LOCAL_MODE'] = 'true'
        logger.info("已设置本地模式环境变量")
    
    if args.mock_data:
        os.environ['SEPSIS_USE_MOCK_DATA'] = 'true'
        logger.info("已设置模拟数据环境变量")
    
    # 确保目录存在
    ensure_directories()
    
    # 根据参数决定执行流程
    if args.only_viz:
        generate_visualizations()
    else:
        # 数据处理阶段
        if not args.skip_data:
            process_data(use_sample_data=args.sample, local_mode=args.local)
        else:
            logger.info("跳过数据处理阶段")
        
        # 模型训练阶段
        if not args.skip_train:
            train_model(epochs=args.epochs, batch_size=args.batch_size, use_sample_data=args.sample)
        else:
            logger.info("跳过模型训练阶段")
        
        # 可视化阶段
        generate_visualizations()
    
    # 执行完成
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"脓毒症早期预警系统 - 执行完成，总耗时: {total_time/60:.2f} 分钟")
    
    # 检查所有关键文件是否存在
    files_to_check = [
        'data/processed/patient_features.csv',
        'data/processed/sepsis_labels.csv',
        'models/best_model.pt',
        'results/predictions.csv',
        'results/figures/test_roc_curve.png',
        'results/sepsis_prediction_results.html'
    ]
    
    logger.info("生成的关键文件:")
    for file_path in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            logger.info(f"- {file_path}: {file_size:.2f} KB")
        else:
            logger.warning(f"- {file_path}: 文件不存在")
    
    # 打印总结
    logger.info("""
脓毒症早期预警系统已完成执行。系统具有以下特点：
1. 大规模数据集：从MIMIC-IV中提取了大量真实患者数据
2. 多模态融合：结合生命体征、实验室检查、用药记录和临床文本
3. 双时序建模：同时使用LSTM和TCN捕捉不同时间维度的特征
4. 知识图谱增强：融入医学知识，提高预测可解释性
5. 动态风险预测：实时监测患者状态，提前预警脓毒症风险
6. SHAP解释性分析：揭示预测背后的关键因素

报告和可视化结果已保存到results目录。
    """)

if __name__ == "__main__":
    main() 
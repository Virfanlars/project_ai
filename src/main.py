#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脓毒症早期预警系统主入口
基于多模态时序数据与知识图谱增强的脓毒症早期预警系统
"""

import argparse
import torch
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor.data_loader import load_data
from src.data_processor.data_imputer import (
    simple_imputation,
    knn_imputation,
    mice_imputation,
    forward_fill_imputation,
    feature_selection
)
from src.models.multimodal_transformer import SepsisTransformerModel
from src.knowledge_graph.kg_builder import build_knowledge_graph
from src.knowledge_graph.kg_embedder import generate_embeddings
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.visualization.visualizer import plot_results
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脓毒症早期预警系统')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='每个数据集使用的最大样本数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 模型相关参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    
    # 知识图谱相关参数
    parser.add_argument('--kg_method', type=str, default='TransE',
                       help='知识图谱嵌入方法')
    parser.add_argument('--kg_embedding_dim', type=int, default=64,
                       help='知识图谱嵌入维度')
    
    # 插补相关参数
    parser.add_argument('--imputation_method', type=str, default=None,
                       help='数据插补方法，可选值: simple, knn, mice, forward_fill, None')
    parser.add_argument('--imputation_strategy', type=str, default='mean',
                       help='简单插补策略，可选值: mean, median, most_frequent')
    parser.add_argument('--knn_neighbors', type=int, default=5,
                       help='KNN插补的邻居数')
    
    # 日志相关参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='日志级别')

    # 流程控制参数
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练阶段，直接使用现有模型进行评估')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='跳过评估阶段')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='跳过可视化阶段')
    parser.add_argument('--model_path', type=str, default=None,
                       help='已有模型路径，用于跳过训练时加载')
    
    return parser.parse_args()

def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        setup_logger(args.log_level, os.path.join(output_dir, 'run.log'))
        logger = logging.getLogger(__name__)
        logger.info(f"启动脓毒症早期预警系统，参数: {vars(args)}")
        
        # 检查设备可用性
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU替代")
            args.device = 'cpu'
        
        device = torch.device(args.device)
        
        # 检查数据目录
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"数据目录 {args.data_dir} 不存在")
            
        # 加载数据
        logger.info("加载数据...")
        try:
            data_loaders, feature_dims = load_data(args.data_dir, args.batch_size, args.max_samples)
            
            # 验证数据加载器
            if not all(k in data_loaders for k in ['train', 'val', 'test']):
                raise ValueError("数据加载器缺少必要的部分(train/val/test)")
                
            # 验证每个数据集都有数据
            for split, loader in data_loaders.items():
                if len(loader.dataset) == 0:
                    raise ValueError(f"{split} 数据集为空")
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
        
        # 应用数据插补（如果指定了方法）
        if args.imputation_method:
            logger.info(f"执行{args.imputation_method}数据插补...")
            
            try:
                for split, loader in data_loaders.items():
                    dataset = loader.dataset
                    
                    # 获取数据
                    vitals_data = dataset.vitals
                    labs_data = dataset.labs
                    
                    # 执行插补
                    for i in range(len(vitals_data)):
                        if args.imputation_method == 'simple':
                            vitals_data[i] = simple_imputation(
                                pd.DataFrame(vitals_data[i]), 
                                method=args.imputation_strategy
                            ).values
                            labs_data[i] = simple_imputation(
                                pd.DataFrame(labs_data[i]), 
                                method=args.imputation_strategy
                            ).values
                            
                        elif args.imputation_method == 'knn':
                            vitals_data[i] = knn_imputation(
                                pd.DataFrame(vitals_data[i]), 
                                n_neighbors=args.knn_neighbors
                            ).values
                            labs_data[i] = knn_imputation(
                                pd.DataFrame(labs_data[i]), 
                                n_neighbors=args.knn_neighbors
                            ).values
                            
                        elif args.imputation_method == 'mice':
                            vitals_data[i] = mice_imputation(
                                pd.DataFrame(vitals_data[i])
                            ).values
                            labs_data[i] = mice_imputation(
                                pd.DataFrame(labs_data[i])
                            ).values
                            
                        elif args.imputation_method == 'forward_fill':
                            vitals_data[i] = forward_fill_imputation(
                                pd.DataFrame(vitals_data[i])
                            ).values
                            labs_data[i] = forward_fill_imputation(
                                pd.DataFrame(labs_data[i])
                            ).values
                            
                        elif args.imputation_method == 'feature_selection':
                            # 特征选择需要特殊处理，因为它会改变特征维度
                            logger.warning("特征选择方法不适用于主程序中内联处理，请使用单独的run_imputation.py脚本")
                    
                    logger.info(f"完成{split}数据集的{args.imputation_method}插补")
                
            except Exception as e:
                logger.error(f"数据插补失败: {e}")
                raise
        
        # 构建知识图谱
        logger.info("构建知识图谱...")
        try:
            kg = build_knowledge_graph(data_loaders['train'].dataset)
            
            # 检查知识图谱是否为空
            if len(kg.entity2id) == 0 or len(kg.relation2id) == 0:
                logger.warning("构建的知识图谱为空，将使用简化模型")
                kg_empty = True
            else:
                kg_empty = False
        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            raise
        
        # 生成知识图谱嵌入
        try:
            if not kg_empty:
                logger.info(f"生成知识图谱嵌入，使用 {args.kg_method} 方法...")
                kg_embeddings = generate_embeddings(kg, args.kg_method, args.kg_embedding_dim, output_dir=output_dir)
            else:
                # 创建空的嵌入
                logger.warning("使用随机初始化的知识图谱嵌入")
                kg_embeddings = {
                    'entity_embeddings': torch.randn(10, args.kg_embedding_dim).numpy(),
                    'relation_embeddings': torch.randn(5, args.kg_embedding_dim).numpy(),
                    'entity2id': {'dummy': 0},
                    'id2entity': {0: 'dummy'}
                }
        except Exception as e:
            logger.error(f"知识图谱嵌入生成失败: {e}")
            raise
        
        # 如果指定了模型路径，则直接加载模型
        if args.skip_training and args.model_path:
            logger.info(f"跳过训练，直接加载模型: {args.model_path}")
            try:
                model = SepsisTransformerModel(
                    vitals_dim=feature_dims['vitals'],
                    lab_dim=feature_dims['lab'],
                    drug_dim=feature_dims['drug'],
                    text_dim=feature_dims['text'],
                    kg_dim=args.kg_embedding_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                ).to(device)
                
                model.load_state_dict(torch.load(args.model_path, map_location=device))
                logger.info("模型加载成功")
                
                # 保存模型配置
                model_config = {
                    'vitals_dim': feature_dims['vitals'],
                    'lab_dim': feature_dims['lab'],
                    'drug_dim': feature_dims['drug'],
                    'text_dim': feature_dims['text'],
                    'kg_dim': args.kg_embedding_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout
                }
                
                with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
                    json.dump(model_config, f, indent=2)
                
                # 设置模型路径
                best_model_path = args.model_path
                
                # 创建空的训练历史
                history = {
                    'train_loss': [],
                    'val_loss': [],
                    'train_auroc': [],
                    'val_auroc': []
                }
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        else:
            # 创建并训练模型
            logger.info("创建多模态Transformer模型...")
            try:
                model = SepsisTransformerModel(
                    vitals_dim=feature_dims['vitals'],
                    lab_dim=feature_dims['lab'],
                    drug_dim=feature_dims['drug'],
                    text_dim=feature_dims['text'],
                    kg_dim=args.kg_embedding_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                ).to(device)
                
                # 确保模型能处理小批量样本
                logger.info("验证模型正确性...")
                batch = next(iter(data_loaders['train']))
                vitals = batch['vitals'][:2].to(device)
                labs = batch['labs'][:2].to(device)
                drugs = batch['drugs'][:2].to(device)
                time_indices = batch['time_indices'][:2].to(device)
                
                # 创建文本和知识图谱嵌入
                text_embeds = torch.zeros(2, 768).to(device)
                kg_embeds = torch.tensor(kg_embeddings['entity_embeddings'][:2], dtype=torch.float32).to(device)
                
                # 创建注意力掩码 - 改进的掩码条件
                # 不仅仅检查所有特征是否为零，而是检查是否有缺失值（NaN或无穷大）
                vitals_valid = ~torch.isnan(vitals).any(dim=2) & ~torch.isinf(vitals).any(dim=2)
                labs_valid = ~torch.isnan(labs).any(dim=2) & ~torch.isinf(labs).any(dim=2)
                drugs_valid = ~torch.isnan(drugs).any(dim=2) & ~torch.isinf(drugs).any(dim=2)
                
                # 此外，检查特征是否全为零
                vitals_nonzero = vitals.abs().sum(dim=2) > 1e-6
                labs_nonzero = labs.abs().sum(dim=2) > 1e-6
                drugs_nonzero = drugs.abs().sum(dim=2) > 1e-6
                
                # 特征要么有效（非NaN和非Inf），要么至少有一个非零值
                valid_positions = (vitals_valid & vitals_nonzero) | (labs_valid & labs_nonzero) | (drugs_valid & drugs_nonzero)
                
                # 反转为掩码（True表示填充位置）
                attention_mask = ~valid_positions
                
                # 确保每个样本至少有一个非掩码位置
                for i in range(attention_mask.size(0)):
                    if attention_mask[i].all():
                        # 对全部被掩码的样本，将第一个位置设为非掩码
                        attention_mask[i, 0] = False
                        # 同时确保相应位置的特征至少有一些有意义的值
                        if i < vitals.size(0) and 0 < vitals.size(1):
                            # 对生命体征赋予合理的默认值
                            vitals[i, 0, 0] = 75.0  # 正常心率约75次/分钟
                            if vitals.size(2) > 1:
                                vitals[i, 0, 1] = 16.0  # 正常呼吸频率约16次/分钟
                            if vitals.size(2) > 2:
                                vitals[i, 0, 2] = 36.8  # 正常体温约36.8°C
                
                # 测试前向传播
                with torch.no_grad():
                    outputs = model(vitals, labs, drugs, text_embeds, kg_embeds, time_indices, attention_mask)
                    logger.info(f"模型前向传播测试成功，输出形状: {outputs.shape}")
            except Exception as e:
                logger.error(f"模型创建或验证失败: {e}")
                raise
            
            # 训练模型
            if not args.skip_training:
                logger.info("开始训练模型...")
                try:
                    history = train_model(
                        model=model,
                        data_loaders=data_loaders,
                        kg_embeddings=kg_embeddings,
                        device=device,
                        learning_rate=args.lr,
                        epochs=args.epochs,
                        patience=args.patience,
                        output_dir=output_dir
                    )
                    
                    # 检查训练是否完成
                    if len(history['train_loss']) == 0:
                        raise RuntimeError("训练未完成，没有产生任何训练记录")
                        
                    # 设置最佳模型路径
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                except Exception as e:
                    logger.error(f"模型训练失败: {e}")
                    raise
            else:
                logger.info("跳过训练阶段")
                # 保存当前模型作为最佳模型
                best_model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                
                # 创建空的训练历史
                history = {
                    'train_loss': [],
                    'val_loss': [],
                    'train_auroc': [],
                    'val_auroc': []
                }
        
        # 评估模型
        metrics = None
        if not args.skip_evaluation:
            logger.info("评估模型...")
            try:
                # 检查最佳模型文件是否存在
                if not os.path.exists(best_model_path):
                    logger.warning("最佳模型文件不存在，使用当前模型进行评估")
                    torch.save(model.state_dict(), best_model_path)
                    
                metrics = evaluate_model(
                    model=model,
                    model_path=best_model_path,
                    data_loader=data_loaders['test'],
                    kg_embeddings=kg_embeddings,
                    device=device,
                    output_dir=output_dir
                )
                
                # 检查指标是否有效
                if metrics is None or len(metrics) == 0:
                    logger.warning("评估未产生有效指标")
                    metrics = {
                        'auroc': 0.5,
                        'auprc': 0.5,
                        'accuracy': 0.5,
                        'sensitivity': 0.5,
                        'specificity': 0.5,
                    }
            except Exception as e:
                logger.error(f"模型评估失败: {e}")
                raise
        else:
            logger.info("跳过评估阶段")
            metrics = None
        
        # 可视化结果
        if not args.skip_visualization and metrics is not None:
            logger.info("生成结果可视化...")
            try:
                plot_results(history, metrics, output_dir)
            except Exception as e:
                logger.error(f"结果可视化失败: {e}")
                raise
        elif args.skip_visualization:
            logger.info("跳过可视化阶段")
        elif metrics is None:
            logger.warning("没有评估指标，跳过可视化")
            
        logger.info(f"脓毒症早期预警系统运行完成，结果保存在 {output_dir}")
        
    except Exception as e:
        logging.error(f"系统运行出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
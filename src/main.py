#!/usr/bin/env python3
"""
主程序：脓毒症风险预测多模态模型
整合外部医学知识库
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# 添加src目錄到Python路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# 導入自定義模塊
from data_processor.data_loader import SepsisDataLoader
from models.multimodal_transformer import SepsisTransformerModel
from training.trainer import SepsisTrainer
from knowledge_graph.kg_builder import KnowledgeGraphBuilder
from knowledge_graph.kg_embedder import KnowledgeGraphEmbedder
from utils.logger import setup_logger

def load_config():
    """
    加载配置文件，包括API密钥
    
    Returns:
        配置字典
    """
    config = {
        'umls_api_key': None,
        'use_external_apis': True,
        'cache_duration_hours': 24,
        'max_api_calls_per_minute': 60
    }
    
    # 尝试从环境变量加载API密钥
    config['umls_api_key'] = os.getenv('UMLS_API_KEY')
    
    # 尝试从配置文件加载
    config_file = Path('config.json')
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logging.warning(f"读取配置文件失败: {e}")
    
    return config

def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "umls_api_key": "YOUR_UMLS_API_KEY_HERE",
        "use_external_apis": True,
        "cache_duration_hours": 24,
        "max_api_calls_per_minute": 60,
        "note": "请在https://uts.nlm.nih.gov/uts/申请UMLS API密钥并替换YOUR_UMLS_API_KEY_HERE"
    }
    
    config_file = Path('config.json')
    if not config_file.exists():
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            print(f"已创建示例配置文件: {config_file}")
            print("请编辑config.json文件，添加您的UMLS API密钥")
        except Exception as e:
            print(f"创建配置文件失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脓毒症风险预测多模态模型')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                       help='运行模式')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--create-config', action='store_true',
                       help='创建示例配置文件')
    parser.add_argument('--skip-kg', action='store_true',
                       help='跳过知识图谱构建（用于测试）')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='数据目录路径')
    parser.add_argument('--output-path', type=str, default='output',
                       help='输出目录路径')
    parser.add_argument('--model-path', type=str, default=None,
                       help='预训练模型路径（用于评估模式）')
    
    args = parser.parse_args()
    
    # 如果用户要求创建配置文件
    if args.create_config:
        create_sample_config()
        return
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_path) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        log_level="INFO",
        log_file=output_dir / "run.log"
    )
    
    logger.info("=" * 60)
    logger.info("脓毒症风险预测多模态模型")
    logger.info("=" * 60)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 加载配置
        config = load_config()
        logger.info(f"配置加载完成，使用外部API: {config['use_external_apis']}")
        
        if config['use_external_apis'] and not config['umls_api_key']:
            logger.warning("未找到UMLS API密钥，将使用受限的功能")
            logger.warning("请运行 --create-config 创建配置文件并添加API密钥")
        
        # 1. 数据加载
        logger.info("开始加载数据...")
        data_loader = SepsisDataLoader(args.data_path)
        
        # 加載和預處理數據
        train_data, val_data, test_data = data_loader.load_and_split_data()
        
        if train_data is None:
            raise RuntimeError("數據加載失敗")
        
        logger.info(f"數據加載完成:")
        logger.info(f"  訓練集: {len(train_data)} 樣本")
        logger.info(f"  驗證集: {len(val_data)} 樣本") 
        logger.info(f"  測試集: {len(test_data)} 樣本")
        
        # 2. 構建知識圖譜
        kg_path = output_dir / "knowledge_graph.pkl"
        
        if not args.skip_kg:
            logger.info("開始構建醫學知識圖譜...")
            
            try:
                kg_builder = KnowledgeGraphBuilder(
                    umls_api_key=config.get('umls_api_key')
                )
                
                # 嘗試加載已有的知識圖譜
                if kg_builder.load_graph(kg_path):
                    logger.info("成功加載已有的知識圖譜")
                    knowledge_graph = kg_builder.graph
                else:
                    logger.info("構建新的知識圖譜...")
                    knowledge_graph = kg_builder.build_sepsis_knowledge_graph()
                    
                    # 保存知識圖譜
                    kg_builder.save_graph(kg_path)
                
                # 獲取圖統計信息
                stats = kg_builder.get_graph_statistics()
                logger.info(f"知識圖譜統計: {stats}")
                
                # 3. 訓練知識圖譜嵌入
                logger.info("開始訓練知識圖譜嵌入...")
                kg_embedder = KnowledgeGraphEmbedder(knowledge_graph)
                embeddings = kg_embedder.train_embeddings()
                
                # 保存嵌入
                embedding_path = output_dir / "kg_embeddings.pkl"
                kg_embedder.save_embeddings(embedding_path)
                logger.info(f"知識圖譜嵌入已保存到: {embedding_path}")
                
            except Exception as e:
                logger.error(f"知識圖譜構建失敗: {e}")
                logger.error("程序將退出")
                raise
        else:
            logger.info("跳過知識圖譜構建")
            knowledge_graph = None
            embeddings = None
        
        # 4. 模型初始化
        logger.info("初始化多模態變換器模型...")
        
        # 獲取特徵維度
        sample_data = next(iter(train_data))
        if isinstance(sample_data, (list, tuple)):
            sample_features = sample_data[0]
        else:
            sample_features = sample_data['features']
        
        input_dim = sample_features.shape[-1] if hasattr(sample_features, 'shape') else len(sample_features)
        
        model = SepsisTransformerModel(
            input_dim=input_dim,
            d_model=256,
            nhead=8,
            num_layers=6,
            knowledge_graph=knowledge_graph,
            kg_embeddings=embeddings
        )
        
        logger.info(f"模型初始化完成，輸入維度: {input_dim}")
        
        # 5. 訓練或評估
        trainer = SepsisTrainer(
            model=model,
            train_loader=train_data,
            val_loader=val_data,
            test_loader=test_data,
            output_dir=output_dir,
            knowledge_graph=knowledge_graph
        )
        
        if args.mode == 'train':
            logger.info("開始模型訓練...")
            
            # 訓練模型
            training_history = trainer.train(
                epochs=50,
                learning_rate=1e-4,
                batch_size=32,
                patience=10
            )
            
            # 保存訓練歷史
            history_path = output_dir / "training_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(training_history, f, indent=2, ensure_ascii=False)
            
            logger.info("模型訓練完成")
            
            # 在測試集上評估
            logger.info("在測試集上評估模型...")
            test_metrics = trainer.evaluate_model()
            
            # 保存評估結果
            metrics_path = output_dir / "evaluation_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"評估完成，結果已保存到: {metrics_path}")
            
        elif args.mode == 'evaluate':
            if args.model_path:
                logger.info(f"加載預訓練模型: {args.model_path}")
                trainer.load_model(args.model_path)
            else:
                logger.error("評估模式需要提供 --model-path 參數")
                return
            
            logger.info("開始模型評估...")
            test_metrics = trainer.evaluate_model()
            
            # 保存評估結果
            eval_dir = output_dir / f"eval_{timestamp}"
            eval_dir.mkdir(exist_ok=True)
            
            metrics_path = eval_dir / "evaluation_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"評估完成，結果已保存到: {metrics_path}")
        
        logger.info("程序執行完成")
        logger.info(f"所有輸出文件保存在: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("用戶中斷程序執行")
    except Exception as e:
        logger.error(f"程序執行過程中發生錯誤: {e}")
        import traceback
        logger.error(f"錯誤詳情:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
"""
更新特征重要性数据并保存到evaluation_results.json文件
"""

import os
import sys
import json
import torch
import numpy as np
import logging

# 设置Python模块搜索路径
sys.path.append(".")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def update_feature_importance():
    """计算特征重要性并更新评估结果文件"""
    
    try:
        # 导入必要的模块
        from models.fusion.multimodal_model import SepsisTransformerModel
        from utils.evaluation import calculate_feature_importance
        from config import MODEL_CONFIG, TRAIN_CONFIG
        
        logger.info("正在加载模型和测试数据...")
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() and TRAIN_CONFIG.get('device') == 'cuda' else 'cpu')
        model = SepsisTransformerModel(
            vitals_dim=MODEL_CONFIG['vitals_dim'],
            lab_dim=MODEL_CONFIG['lab_dim'],
            drug_dim=MODEL_CONFIG['drug_dim'],
            text_dim=MODEL_CONFIG['text_dim'],
            kg_dim=MODEL_CONFIG['kg_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        )
        
        # 加载训练好的模型参数
        model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
        logger.info("模型加载成功")
        
        # 加载数据集划分信息
        dataset_splits = torch.load('models/dataset_splits.pt')
        test_indices = dataset_splits['test_indices']
        
        # 准备测试数据加载器
        logger.info("准备测试数据加载器...")
        from run_visualization import prepare_test_loader
        test_loader = prepare_test_loader(test_indices)
        
        # 定义损失函数
        criterion = torch.nn.BCELoss()
        
        # 计算特征重要性
        logger.info("计算特征重要性...")
        feature_importance = calculate_feature_importance(model, test_loader, device, criterion)
        
        # 打印特征重要性排名前5的特征
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("特征重要性前5名:")
        for feature, importance in sorted_importance[:5]:
            logger.info(f"{feature}: {importance:.4f}")
        
        # 加载现有评估结果
        eval_results_path = 'results/evaluation_results.json'
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
            
            # 添加特征重要性到评估结果
            eval_results['feature_importance'] = feature_importance
            
            # 保存更新后的评估结果
            with open(eval_results_path, 'w') as f:
                json.dump(eval_results, f, indent=4)
                
            logger.info(f"特征重要性已添加到 {eval_results_path}")
        else:
            logger.error(f"评估结果文件不存在: {eval_results_path}")
            
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"更新特征重要性时出错: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = update_feature_importance()
    if success:
        print("特征重要性更新成功！")
    else:
        print("特征重要性更新失败，请查看日志了解详情。") 
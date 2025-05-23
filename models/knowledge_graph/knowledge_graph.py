import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.nn import RGCNConv
import numpy as np

# 构建知识图谱
def build_medical_kg():
    """
    构建医学知识图谱
    包含疾病、药物和实验室检查项三种节点类型
    以及它们之间的关系
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    # 1. 疾病节点 (ICD编码)
    diseases = [
        ('D001', 'Sepsis'), 
        ('D002', 'Pneumonia'),
        ('D003', 'Urinary Tract Infection'),
        ('D004', 'Bacteremia'),
        ('D005', 'Meningitis'),
        ('D006', 'Cellulitis'),
        ('D007', 'Endocarditis'),
        ('D008', 'Osteomyelitis'),
        ('D009', 'Peritonitis'),
        ('D010', 'Pyelonephritis')
    ]
    
    # 2. 药物节点
    drugs = [
        ('M001', 'Ceftriaxone'),
        ('M002', 'Vancomycin'),
        ('M003', 'Piperacillin-Tazobactam'),
        ('M004', 'Meropenem'),
        ('M005', 'Ciprofloxacin'),
        ('M006', 'Metronidazole'),
        ('M007', 'Azithromycin'),
        ('M008', 'Levofloxacin'),
        ('M009', 'Cefepime'),
        ('M010', 'Ampicillin-Sulbactam')
    ]
    
    # 3. 实验室检查节点
    labs = [
        ('L001', 'WBC'),
        ('L002', 'Lactate'),
        ('L003', 'Creatinine'),
        ('L004', 'Platelet'),
        ('L005', 'Bilirubin')
    ]
    
    # 添加所有节点
    for node_id, name in diseases + drugs + labs:
        G.add_node(node_id, name=name, 
                  type='disease' if node_id.startswith('D') else
                       'drug' if node_id.startswith('M') else 'lab')
    
    # 添加边
    # 1. 疾病-药物关系 (治疗关系)
    disease_drug_edges = [
        ('D001', 'M003', 'treats'),  # Sepsis - Piperacillin-Tazobactam
        ('D001', 'M004', 'treats'),  # Sepsis - Meropenem
        ('D002', 'M001', 'treats'),  # Pneumonia - Ceftriaxone
        ('D002', 'M007', 'treats'),  # Pneumonia - Azithromycin
        ('D003', 'M005', 'treats'),  # UTI - Ciprofloxacin
        ('D004', 'M002', 'treats'),  # Bacteremia - Vancomycin
        ('D005', 'M001', 'treats'),  # Meningitis - Ceftriaxone
        ('D006', 'M002', 'treats'),  # Cellulitis - Vancomycin
        ('D007', 'M002', 'treats'),  # Endocarditis - Vancomycin
        ('D008', 'M002', 'treats'),  # Osteomyelitis - Vancomycin
        ('D009', 'M006', 'treats'),  # Peritonitis - Metronidazole
        ('D010', 'M005', 'treats'),  # Pyelonephritis - Ciprofloxacin
    ]
    
    # 2. 疾病-实验室检查关系 (诊断指标)
    disease_lab_edges = [
        ('D001', 'L001', 'diagnoses_with'),  # Sepsis - WBC
        ('D001', 'L002', 'diagnoses_with'),  # Sepsis - Lactate
        ('D002', 'L001', 'diagnoses_with'),  # Pneumonia - WBC
        ('D003', 'L001', 'diagnoses_with'),  # UTI - WBC
        ('D003', 'L003', 'diagnoses_with'),  # UTI - Creatinine
        ('D004', 'L001', 'diagnoses_with'),  # Bacteremia - WBC
        ('D005', 'L001', 'diagnoses_with'),  # Meningitis - WBC
        ('D006', 'L001', 'diagnoses_with'),  # Cellulitis - WBC
        ('D007', 'L001', 'diagnoses_with'),  # Endocarditis - WBC
        ('D008', 'L001', 'diagnoses_with'),  # Osteomyelitis - WBC
        ('D009', 'L001', 'diagnoses_with'),  # Peritonitis - WBC
        ('D010', 'L001', 'diagnoses_with'),  # Pyelonephritis - WBC
        ('D010', 'L003', 'diagnoses_with'),  # Pyelonephritis - Creatinine
    ]
    
    # 3. 药物-药物交互关系
    drug_drug_edges = [
        ('M001', 'M007', 'synergizes_with'),  # Ceftriaxone - Azithromycin
        ('M003', 'M006', 'synergizes_with'),  # Piperacillin-Tazobactam - Metronidazole
        ('M002', 'M006', 'synergizes_with'),  # Vancomycin - Metronidazole
    ]
    
    # 4. 实验室检查-实验室检查关系（相关性）
    lab_lab_edges = [
        ('L001', 'L002', 'correlates_with'),  # WBC - Lactate
        ('L003', 'L005', 'correlates_with'),  # Creatinine - Bilirubin
        ('L004', 'L005', 'correlates_with'),  # Platelet - Bilirubin
    ]
    
    # 添加所有边
    for src, dst, relation in disease_drug_edges + disease_lab_edges + drug_drug_edges + lab_lab_edges:
        G.add_edge(src, dst, relation=relation)
    
    return G

# 转换为PyTorch Geometric格式
def convert_nx_to_pytorch_geometric(G):
    """
    将NetworkX图转换为PyTorch Geometric格式
    """
    # 节点类型映射
    node_types = {'disease': 0, 'drug': 1, 'lab': 2}
    
    # 关系类型映射
    relation_types = {
        'treats': 0, 
        'diagnoses_with': 1, 
        'synergizes_with': 2, 
        'correlates_with': 3
    }
    
    # 节点特征矩阵 (one-hot编码)
    x = torch.eye(len(G.nodes))
    
    # 边和关系类型
    edge_index = []
    edge_type = []
    
    # 节点ID映射
    node_id_map = {node: i for i, node in enumerate(G.nodes)}
    
    for src, dst, data in G.edges(data=True):
        edge_index.append([node_id_map[src], node_id_map[dst]])
        edge_type.append(relation_types[data['relation']])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_type = torch.tensor(edge_type)
    
    # 节点类型
    node_type = torch.tensor([
        node_types[G.nodes[node]['type']] for node in G.nodes
    ])
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type)

# 生成知识图谱嵌入
def generate_kg_embeddings(pyg_data, embedding_dim=64, train_model=False, epochs=30):
    """
    生成知识图谱嵌入
    使用关系图卷积网络(RGCN)
    
    参数:
    - pyg_data: PyTorch Geometric格式的图数据
    - embedding_dim: 嵌入维度
    - train_model: 是否训练模型(默认为False，直接使用随机初始化)
    - epochs: 训练轮数(仅当train_model为True时有效)
    
    返回:
    - node_embeddings: 节点嵌入向量
    """
    # 设置为评估模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pyg_data = pyg_data.to(device)
    
    # 提取参数
    num_nodes = pyg_data.x.size(0)
    num_relations = len(torch.unique(pyg_data.edge_type))
    num_node_types = len(torch.unique(pyg_data.node_type))
    
    # 创建RGCN模型
    class RGCN(torch.nn.Module):
        def __init__(self):
            super(RGCN, self).__init__()
            self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
            
            # 增加节点类型嵌入
            self.type_embedding = torch.nn.Embedding(num_node_types, embedding_dim)
            
            # 多层RGCN结构(使用num_bases减少参数量)
            self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=8)
            self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=8)
            
            # 添加层归一化
            self.norm1 = torch.nn.LayerNorm(embedding_dim)
            self.norm2 = torch.nn.LayerNorm(embedding_dim)
            
            # 添加Dropout防止过拟合
            self.dropout = torch.nn.Dropout(0.2)
            
        def forward(self, x, edge_index, edge_type, node_type=None):
            # 初始嵌入
            x = self.embedding.weight
            
            # 融合节点类型信息(如果提供)
            if node_type is not None:
                type_emb = self.type_embedding(node_type)
                x = x + type_emb
            
            # 第一层卷积 + 残差
            identity = x
            x = self.conv1(x, edge_index, edge_type)
            x = self.norm1(x)
            x = torch.relu(x)
            x = self.dropout(x)
            
            # 第二层卷积 + 残差
            x = self.conv2(x, edge_index, edge_type)
            x = x + identity  # 添加残差连接
            x = self.norm2(x)
            x = torch.relu(x)
            
            return x
    
    # 初始化模型
    model = RGCN().to(device)
    
    # 如果需要训练模型
    if train_model:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 创建边预测任务的损失函数
        def link_prediction_loss(pos_edge_index, neg_edge_index, embeddings):
            # 正样本分数
            src_pos, dst_pos = pos_edge_index
            pos_scores = torch.sum(embeddings[src_pos] * embeddings[dst_pos], dim=1)
            
            # 负样本分数
            src_neg, dst_neg = neg_edge_index
            neg_scores = torch.sum(embeddings[src_neg] * embeddings[dst_neg], dim=1)
            
            # 使用margin ranking loss
            margin = 0.1
            return torch.nn.functional.margin_ranking_loss(
                pos_scores, neg_scores, 
                torch.ones_like(pos_scores, device=device), 
                margin=margin
            )
        
        # 获取原始边
        edge_index = pyg_data.edge_index
        
        # 训练循环
        print(f"开始训练知识图谱嵌入模型，共 {epochs} 轮...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 获取节点嵌入
            embeddings = model(pyg_data.x, edge_index, pyg_data.edge_type, pyg_data.node_type)
            
            # 创建正样本
            # 随机选择80%的边作为正样本
            perm = torch.randperm(edge_index.size(1), device=device)
            train_idx = perm[:int(0.8 * edge_index.size(1))]
            pos_edge_index = edge_index[:, train_idx]
            
            # 创建负样本（随机节点对）
            src_nodes = pos_edge_index[0]
            dst_nodes = torch.randint(0, num_nodes, (len(src_nodes),), device=device)
            neg_edge_index = torch.stack([src_nodes, dst_nodes])
            
            # 计算损失
            loss = link_prediction_loss(pos_edge_index, neg_edge_index, embeddings)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'轮次 {epoch+1}/{epochs}, 损失: {loss.item():.4f}')
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        node_embeddings = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, pyg_data.node_type)
    
    return node_embeddings.cpu()
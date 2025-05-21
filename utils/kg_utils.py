import os
import torch
import pickle
from models.knowledge_graph.knowledge_graph import build_medical_kg, convert_nx_to_pytorch_geometric, generate_kg_embeddings

def get_kg_embeddings(embedding_dim=64, force_rebuild=False, cache_dir='data/kg_cache'):
    """
    获取知识图谱嵌入，优先使用缓存
    
    参数:
    - embedding_dim: 嵌入维度
    - force_rebuild: 是否强制重建嵌入，忽略缓存
    - cache_dir: 缓存目录
    
    返回:
    - node_embeddings: 节点嵌入
    - node_id_map: 节点ID到索引的映射
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'kg_emb_{embedding_dim}.pkl')
    
    # 检查缓存
    if not force_rebuild and os.path.exists(cache_file):
        print(f"Loading KG embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            return cache_data['embeddings'], cache_data['node_id_map']
    
    # 重建知识图谱和嵌入
    print("Building medical knowledge graph...")
    kg = build_medical_kg()
    
    # 获取节点ID映射
    node_id_map = {node: i for i, node in enumerate(kg.nodes)}
    
    # 转换为PyG格式
    print("Converting to PyTorch Geometric format...")
    pyg_data = convert_nx_to_pytorch_geometric(kg)
    
    # 生成嵌入
    print(f"Generating KG embeddings (dim={embedding_dim})...")
    node_embeddings = generate_kg_embeddings(
        pyg_data, 
        embedding_dim=embedding_dim,
        train_model=True,  # 训练模型以获得更好的嵌入
        epochs=50
    )
    
    # 缓存结果
    cache_data = {
        'embeddings': node_embeddings,
        'node_id_map': node_id_map
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"KG embeddings saved to: {cache_file}")
    return node_embeddings, node_id_map
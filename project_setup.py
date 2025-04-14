# 基础环境配置
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, RGCNConv
from transformers import BertModel, BertTokenizer

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# MIMIC-IV访问配置
from mimic_utils import connect_to_mimic_db
conn = connect_to_mimic_db(user='username', password='password') 
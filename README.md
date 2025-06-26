# 脓毒症风险预测多模态模型

基于多模态变换器和外部医学知识库的脓毒症风险预测系统，整合了UMLS、Clinical Tables、RxNorm等真实医学数据库。

## 功能特性

- **多模态数据融合**: 整合时间序列数据、实验室检查和生命体征
- **外部知识库集成**: 连接UMLS、Clinical Tables、RxNorm等医学数据库
- **知识图谱增强**: 基于真实医学本体构建知识图谱
- **可解释AI**: 提供注意力机制和特征重要性分析
- **实时预测**: 支持实时脓毒症风险评估

## 外部数据库支持

### 已集成并测试的医学数据库

1. **UMLS (统一医学语言系统)** ⚠️ 需要API密钥
   - 统一医学语言系统
   - 提供多种医学词汇的映射和关系
   - 需要从 [UMLS UTS](https://uts.nlm.nih.gov/uts/) 申请API密钥
   - **状态**: 服务可达，需要认证

2. **Clinical Tables (NLM临床数据表)** ✅ 免费可用
   - **Medical Conditions**: 医学条件和疾病 (✅ 已测试)
   - **HPO Phenotypes**: 人类表型本体 (✅ 已测试，23个发热相关结果)
   - **Disease Names**: 疾病名称数据库 (✅ 已测试，4个脓毒症相关结果)
   - **ICD-10-CM**: 国际疾病分类代码 (⚠️ 需要调整搜索词)
   - **状态**: 完全免费，无需认证

3. **RxNorm (标准化药物命名)** ✅ 免费可用
   - 标准化药物命名系统
   - 包含药物相互作用信息
   - **Drug Search**: 药物搜索 (✅ 已测试)
   - **Version Info**: 版本信息 (✅ 2025年6月2日版本)
   - **Approximate Match**: 模糊匹配 (✅ 已测试)
   - **状态**: 完全免费，无需认证

## 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 测试API可用性

首先运行API测试脚本确认所有外部服务正常：

```bash
python test_apis.py
```

### 3. 配置API密钥 (可选)

创建配置文件以获得完整的UMLS功能：

```bash
python src/main.py --create-config
```

编辑生成的 `config.json` 文件：

```json
{
  "umls_api_key": "YOUR_UMLS_API_KEY_HERE",
  "use_external_apis": true,
  "cache_duration_hours": 24,
  "max_api_calls_per_minute": 60,
  "note": "请在https://uts.nlm.nih.gov/uts/申请UMLS API密钥"
}
```

### 4. 获取UMLS API密钥 (可选，但推荐)

1. 访问 [UMLS UTS](https://uts.nlm.nih.gov/uts/)
2. 创建账户并申请API密钥
3. 将密钥添加到配置文件或设置环境变量：
   ```bash
   export UMLS_API_KEY="your_api_key_here"
   ```

**注意**: 即使没有UMLS API密钥，系统仍然可以使用Clinical Tables和RxNorm API正常工作。

## 使用方法

### 训练模式

```bash
# 使用外部知识库训练完整模型
python src/main.py --mode train --data-path data/processed --output-path output

# 跳过知识图谱构建（快速测试）
python src/main.py --mode train --skip-kg --data-path data/processed
```

### 评估模式

```bash
# 评估预训练模型
python src/main.py --mode evaluate --model-path output/run_xxx/best_model.pt --data-path data/processed
```

## 实际API测试结果

基于最新测试（2024年），以下是各API的实际状态：

### ✅ 完全可用的API
- **Clinical Tables**: 医学条件、HPO表型、疾病名称
- **RxNorm**: 药物搜索、版本信息、模糊匹配

### ⚠️ 需要密钥的API
- **UMLS UTS**: 需要免费申请API密钥

### 🔧 需要调整的API
- **某些Clinical Tables子集**: 可能需要调整搜索参数

## 项目结构

```
project/
├── src/
│   ├── main.py                          # 主程序
│   ├── data_processor/
│   │   ├── data_loader.py               # 数据加载器
│   │   └── data_imputer.py              # 数据插补
│   ├── models/
│   │   └── multimodal_transformer.py   # 多模态变换器模型
│   ├── knowledge_graph/
│   │   ├── external_knowledge.py       # 外部知识库集成
│   │   ├── kg_builder.py               # 知识图谱构建器
│   │   └── kg_embedder.py              # 知识图谱嵌入
│   ├── training/
│   │   └── trainer.py                  # 模型训练器
│   ├── evaluation/
│   │   └── evaluator.py                # 模型评估器
│   ├── visualization/
│   │   └── visualizer.py               # 结果可视化
│   └── utils/
│       └── logger.py                   # 日志工具
├── test_apis.py                        # API测试脚本
├── data/                               # 数据目录
├── output/                             # 输出目录
├── config.json                         # 配置文件
└── requirements.txt                    # 依赖列表
```

## 外部知识库功能

### 概念搜索
- ✅ 在Clinical Tables中搜索医学条件和疾病
- ✅ 在HPO中查找人类表型
- ✅ 在UMLS中搜索统一医学概念（需API密钥）

### 药物信息
- ✅ 在RxNorm中搜索标准化药物名称
- ✅ 查询药物相互作用
- ✅ 获取药物版本信息

### 知识图谱构建
- 自动从外部数据库构建脓毒症相关知识图谱
- 特征到医学概念的智能映射
- 支持多源知识整合

## 输出结果

### 模型文件
- `best_model.pt`: 最佳训练模型
- `training_history.json`: 训练历史
- `model_config.json`: 模型配置

### 知识图谱
- `knowledge_graph.pkl`: 构建的知识图谱
- `kg_embeddings.pkl`: 知识图谱嵌入

### 评估结果
- `evaluation_metrics.json`: 详细评估指标
- `figures/`: 可视化图表
- `patient_trajectories.csv`: 患者风险轨迹
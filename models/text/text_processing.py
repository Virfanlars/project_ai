from transformers import BertTokenizer, BertModel
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 处理护理记录文本
def process_nursing_notes(conn, patient_ids):
    query = """
    SELECT subject_id, hadm_id, charttime, text
    FROM noteevents
    WHERE category = 'Nursing' AND subject_id IN %(ids)s
    """
    
    notes_df = pd.read_sql(query, conn, params={'ids': tuple(patient_ids)})
    
    # 按小时汇总文本
    notes_df['charttime'] = pd.to_datetime(notes_df['charttime'])
    notes_df['hour'] = notes_df['charttime'].dt.floor('H')
    
    # 文本预处理
    notes_df['processed_text'] = notes_df['text'].apply(preprocess_text)
    
    return notes_df

# BERT编码
def encode_clinical_text(notes_df, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    text_embeddings = {}
    
    for idx, row in notes_df.iterrows():
        inputs = tokenizer(row['processed_text'], 
                          return_tensors='pt',
                          max_length=max_length,
                          padding='max_length',
                          truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS]标记的最后隐藏状态作为文本表示
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
        key = (row['subject_id'], row['hadm_id'], row['hour'])
        text_embeddings[key] = embedding
    
    return text_embeddings 

# 预处理临床文本
def process_clinical_text(text):
    """
    预处理护理记录文本，支持中英文混合文本
    """
    if not isinstance(text, str):
        return ""
    
    try:
        # 尝试解码可能的编码问题
        if '\\' in text or any(ord(c) > 127 for c in text):
            try:
                # 尝试不同的编码方式
                for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
                    try:
                        decoded = text.encode('latin1').decode(encoding)
                        if all(ord(c) < 65536 for c in decoded):  # 有效的Unicode字符
                            text = decoded
                            break
                    except:
                        continue
            except:
                pass  # 如果所有解码都失败，使用原始文本
    except:
        # 如果处理编码出错，使用原始文本
        pass
    
    # 针对中文特性的处理
    # 中文不需要像英文那样用空格分词，但需要保留标点符号
    if any('\u4e00' <= c <= '\u9fff' for c in text):  # 检测是否包含中文
        # 中文标点符号处理 - 保留中文标点但在两侧增加空格便于分割
        text = re.sub(r'([，。！？；：、])', r' \1 ', text)
    
    # 通用处理：移除特殊字符，但保留中英文字符和常见标点
    text = re.sub(r'[^\w\s.,;:\-\(\)，。！？；：、\u4e00-\u9fff]', ' ', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 提取关键句子（包含疑似感染或脓毒症相关术语的句子）
    infection_terms = [
        # 英文术语
        'infect', 'sepsis', 'septic', 'bacteria', 'fever', 'pneumonia',
        'uti', 'wbc', 'white blood cell', 'antibiotics', 'culture',
        'organism', 'elevated temp', 'high temp', 'lactate',
        # 中文术语
        '感染', '脓毒', '菌血症', '败血症', '细菌', '发热', '肺炎',
        '尿路感染', '白细胞', '抗生素', '培养', '乳酸'
    ]
    
    # 为中英文分别处理句子分割
    # 对于英文使用.!?分割，对于中文使用。！？分割
    sentences = []
    for s in re.split(r'([.!?。！？])', text):
        if s and s not in '.!?。！？':
            sentences.append(s.strip())
    
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if any(term in sentence for term in infection_terms):
            relevant_sentences.append(sentence)
    
    # 如果没有找到相关句子，返回原文本的截断版本
    if not relevant_sentences:
        return text[:512]  # 限制长度
    
    # 合并相关句子
    processed_text = '. '.join(relevant_sentences)
    
    # 限制长度
    return processed_text[:512]

# BERT编码器类
class ClinicalBertEncoder:
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text, max_length=128):
        """编码单个文本"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的最后隐藏状态作为文本表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]  # 返回1D数组

    def __call__(self, text, max_length=128):
        return self.encode(text, max_length)

# 全局编码器实例
_encoder = None

def get_encoder():
    """单例模式获取编码器"""
    global _encoder
    if _encoder is None:
        _encoder = ClinicalBertEncoder()
    return _encoder

def encode_clinical_text(notes_df, max_length=128):
    """
    将护理记录DataFrame中的文本编码为向量
    
    参数:
        notes_df: 包含processed_text列的DataFrame
        max_length: 最大标记长度
        
    返回:
        字典，键为(subject_id, hadm_id, hour)，值为文本嵌入
    """
    encoder = get_encoder()
    text_embeddings = {}
    
    for idx, row in notes_df.iterrows():
        if not isinstance(row['processed_text'], str) or not row['processed_text'].strip():
            embedding = np.zeros(768)  # 空文本返回零向量
        else:
            embedding = encoder(row['processed_text'], max_length)
        
        key = (row['subject_id'], row['hadm_id'], row['hour'])
        text_embeddings[key] = embedding
    
    return text_embeddings 
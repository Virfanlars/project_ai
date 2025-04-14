# 脓毒症标签生成逻辑
import pandas as pd

def calculate_sofa(row):
    sofa_score = 0
    
    # 呼吸系统 - 使用SpO2代替PaO2/FiO2
    if row.get('spo2') is not None:
        if row['spo2'] < 90:
            sofa_score += 1
        if row['spo2'] < 85:
            sofa_score += 1
    
    # 心血管系统 - 血压和血管活性药物
    if row.get('systolic_bp') is not None and row['systolic_bp'] < 100:
        sofa_score += 1
    if any([row.get(v, 0) > 0 for v in ['vasopressor_1', 'vasopressor_2', 'vasopressor_3', 'vasopressor_4']]):
        sofa_score += 2
    
    # 肝脏 - 胆红素
    if row.get('bilirubin') is not None:
        if row['bilirubin'] > 1.2:
            sofa_score += 1
        if row['bilirubin'] > 2:
            sofa_score += 1
    
    # 凝血 - 血小板
    if row.get('platelet') is not None:
        if row['platelet'] < 150:
            sofa_score += 1
        if row['platelet'] < 100:
            sofa_score += 1
    
    # 肾脏 - 肌酐
    if row.get('creatinine') is not None:
        if row['creatinine'] > 1.2:
            sofa_score += 1
        if row['creatinine'] > 2.0:
            sofa_score += 1
    
    return sofa_score

def build_sepsis_labels(aligned_data):
    # 定义脓毒症：SOFA评分增加≥2且存在感染证据（抗生素使用）
    # 为简化，我们假设抗生素使用即表示存在感染
    aligned_data['infection_evidence'] = aligned_data[['antibiotic_1', 'antibiotic_2', 'antibiotic_3', 'antibiotic_4', 'antibiotic_5']].sum(axis=1) > 0
    
    # 计算每个患者基线SOFA评分（取前24小时的最低值）
    baseline_sofa = aligned_data.groupby(['subject_id', 'hadm_id']).apply(
        lambda x: x.sort_values('hour').head(24)['sofa_score'].min()
    ).reset_index().rename(columns={0: 'baseline_sofa'})
    
    # 合并基线SOFA
    aligned_data = pd.merge(aligned_data, baseline_sofa, on=['subject_id', 'hadm_id'])
    
    # 标记脓毒症发生
    aligned_data['sepsis_label'] = ((aligned_data['sofa_score'] - aligned_data['baseline_sofa'] >= 2) & 
                                 aligned_data['infection_evidence']).astype(int)
    
    # 返回标签数据
    return aligned_data[['subject_id', 'hadm_id', 'hour', 'sepsis_label']] 
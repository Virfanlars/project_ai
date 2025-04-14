# 生成MIMIC-IV派生表的脚本
# 该脚本根据MIMIC-IV官方提供的代码创建派生表，如SOFA评分、胆红素等

import pandas as pd
import numpy as np
import os
import sys
import time
import psycopg2

# 添加项目根目录到系统路径，以便正确导入utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
print(f"项目根目录: {project_root}")
print(f"系统路径: {sys.path}")

# 尝试直接导入
try:
    from utils.database_config import get_connection_string, DATABASE_CONFIG
except ImportError:
    # 如果直接导入失败，尝试使用绝对导入
    sys.path.append(os.path.join(project_root, 'utils'))
    from database_config import get_connection_string, DATABASE_CONFIG
    print("使用绝对路径导入成功")

def connect_to_mimic_db():
    """
    连接到MIMIC-IV数据库
    
    返回:
        psycopg2连接对象
    """
    try:
        # 直接使用psycopg2连接
        import psycopg2
        config = DATABASE_CONFIG
        print(f"尝试连接到 {config['host']}:{config['port']}/{config['database']} 数据库...")
        
        # 使用连接字符串
        conn_string = get_connection_string()
        print(f"使用连接字符串: {conn_string}")
        
        # 使用最直接的连接方式
        conn = psycopg2.connect(conn_string)
        
        # 测试连接
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"数据库连接成功! PostgreSQL版本: {version}")
        
        return conn
    except Exception as e:
        print(f"连接数据库失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_derived_schema():
    """创建派生数据schema"""
    conn = None
    try:
        # 创建直接的psycopg2连接以执行DDL语句
        config = DATABASE_CONFIG
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 检查schema是否已存在
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'mimiciv_derived';")
        if cursor.fetchone() is None:
            cursor.execute("CREATE SCHEMA mimiciv_derived;")
            print("已创建mimiciv_derived schema")
        else:
            print("mimiciv_derived schema已存在")
        
        # 授予权限
        cursor.execute(f"GRANT ALL PRIVILEGES ON SCHEMA mimiciv_derived TO {config['user']};")
        
    except Exception as e:
        print(f"创建schema时出错: {e}")
    finally:
        if conn:
            conn.close()

def create_sofa_score_table(conn):
    """创建SOFA评分表"""
    try:
        print("开始创建SOFA评分表...")
        
        # SOFA评分表创建SQL
        sofa_sql = """
        DROP TABLE IF EXISTS mimiciv_derived.sofa;
        CREATE TABLE mimiciv_derived.sofa AS
        -- 定义基础内容
        WITH
        vaso_stg AS (
            SELECT 
                stay_id AS icustay_id, starttime, endtime
                , CASE WHEN itemid IN (221906, 221289) THEN 'norepinephrine'
                      WHEN itemid IN (221662) THEN 'dopamine'
                      WHEN itemid IN (221749) THEN 'epinephrine'
                      WHEN itemid = 222315 THEN 'vasopressin'
                      ELSE NULL END AS vaso_type
                , CASE WHEN itemid IN (221906, 221289) THEN rate -- norepinephrine
                      WHEN itemid = 221662 THEN rate -- dopamine
                      WHEN itemid = 221749 THEN rate -- epinephrine
                      WHEN itemid = 222315 THEN rate -- vasopressin
                      ELSE NULL END AS rate
            FROM mimiciv_icu.inputevents
            WHERE itemid IN
            (
                221906, 221289, -- norepinephrine
                221662, -- dopamine
                221749, -- epinephrine
                222315 -- vasopressin
            )
            AND rate IS NOT NULL
            AND rate > 0
            AND orderid IS NOT NULL
        ),
        vaso_mv AS (
            SELECT
              icustay_id
            , starttime, endtime
            , vaso_type
            , rate AS vaso_rate
            , ROW_NUMBER() OVER (PARTITION BY icustay_id, starttime, vaso_type ORDER BY rate DESC) AS rn
            FROM vaso_stg
        ),
        vaso_cal AS (
            -- 计算最大使用率
            SELECT
                v.icustay_id
              , hr
              , MAX(CASE WHEN vaso_type = 'norepinephrine' THEN vaso_rate ELSE NULL END) AS rate_norepinephrine
              , MAX(CASE WHEN vaso_type = 'dopamine' THEN vaso_rate ELSE NULL END) AS rate_dopamine
              , MAX(CASE WHEN vaso_type = 'epinephrine' THEN vaso_rate ELSE NULL END) AS rate_epinephrine
              , MAX(CASE WHEN vaso_type = 'vasopressin' THEN vaso_rate ELSE NULL END) AS rate_vasopressin
            FROM vaso_mv v
            INNER JOIN mimiciv_icu.icustays ie ON v.icustay_id = ie.stay_id
            -- 对ICU小时逐一计算
            CROSS JOIN generate_series(0, 
                CASE 
                    WHEN EXTRACT(EPOCH FROM ie.outtime - ie.intime) = 0 THEN 0
                    ELSE CEIL(EXTRACT(EPOCH FROM ie.outtime - ie.intime)/60.0/60.0)::INTEGER
                END
            ) AS hr
            WHERE rn = 1
            AND ie.intime + (hr || ' hour')::interval - interval '1 hour' < endtime
            AND ie.intime + (hr || ' hour')::interval > starttime
            GROUP BY v.icustay_id, hr
        ),
        vaso_cv AS (
            SELECT
                icustay_id
              , hr
              -- 计算血管活性药物的分数
              , MAX(
                  CASE 
                    -- 去甲肾上腺素
                    WHEN rate_norepinephrine > 0.1 THEN 4
                    WHEN rate_norepinephrine > 0.05 THEN 3
                    WHEN rate_norepinephrine > 0.01 THEN 2
                    WHEN rate_norepinephrine > 0 THEN 1
                    -- 肾上腺素
                    WHEN rate_epinephrine > 0.1 THEN 4
                    WHEN rate_epinephrine > 0.05 THEN 3
                    WHEN rate_epinephrine > 0.01 THEN 2
                    WHEN rate_epinephrine > 0 THEN 1
                    -- 多巴胺
                    WHEN rate_dopamine > 15 THEN 4
                    WHEN rate_dopamine > 5 THEN 3
                    WHEN rate_dopamine > 2.5 THEN 2
                    WHEN rate_dopamine > 0 THEN 1
                    -- 加压素
                    WHEN rate_vasopressin > 0 THEN 1
                    ELSE 0
                  END) AS vaso_score
            FROM vaso_cal
            GROUP BY icustay_id, hr
        ),
        pafi_stg AS (
            SELECT ie.stay_id AS icustay_id
              , ie.hadm_id
              , bg.charttime
              , po2 AS pao2
              , fio2
              , CASE 
                  WHEN fio2 = 0 OR fio2 IS NULL THEN NULL
                  ELSE (po2/fio2) 
                END AS pao2fio2ratio
              -- 获取对应的ICU小时
              , CEIL(EXTRACT(EPOCH FROM (bg.charttime - ie.intime))/60.0/60.0)::INTEGER AS hr
            FROM mimiciv_icu.icustays ie
            LEFT JOIN mimiciv_icu.chartevents ce
              ON ie.stay_id = ce.stay_id
              AND ce.itemid = 223835 -- FiO2
              AND ce.valuenum > 0 and ce.valuenum <= 100
            LEFT JOIN mimiciv_derived.bg bg
              ON ie.subject_id = bg.subject_id
              AND bg.charttime BETWEEN ie.intime AND ie.outtime
              AND bg.po2 IS NOT NULL
              AND bg.fio2 IS NOT NULL
              AND bg.fio2 > 0  -- 确保fio2大于0
            WHERE ce.itemid IS NOT NULL
              AND bg.po2 IS NOT NULL
              AND bg.fio2 IS NOT NULL
              AND bg.fio2 > 0  -- 再次确保fio2大于0
        ),
        pafi_mv AS (
            -- 获取每小时的最小P/F比
            SELECT icustay_id, hr
              , MIN(pao2fio2ratio) AS pao2fio2ratio_min
            FROM pafi_stg
            GROUP BY icustay_id, hr
        ),
        pafi_cv AS (
            SELECT
                icustay_id
              , hr
              , CASE
                  WHEN pao2fio2ratio_min < 100 THEN 4
                  WHEN pao2fio2ratio_min < 200 THEN 3
                  WHEN pao2fio2ratio_min < 300 THEN 2
                  WHEN pao2fio2ratio_min < 400 THEN 1
                  ELSE 0
                END AS pao2fio2_score
            FROM pafi_mv
        ),
        labs_stg AS (
            -- 所有相关实验室检查
            SELECT 
                ie.stay_id AS icustay_id
              , ie.hadm_id
              , le.charttime
              -- 白细胞
              , CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum ELSE NULL END AS wbc
              -- 血小板
              , CASE WHEN le.itemid = 51265 THEN le.valuenum ELSE NULL END AS platelet
              -- 胆红素
              , CASE WHEN le.itemid = 50885 THEN le.valuenum ELSE NULL END AS bilirubin
              -- 肌酐
              , CASE WHEN le.itemid = 50912 THEN le.valuenum ELSE NULL END AS creatinine
              , CEIL(EXTRACT(EPOCH FROM (le.charttime - ie.intime))/60.0/60.0)::INTEGER AS hr
            FROM mimiciv_icu.icustays ie
            LEFT JOIN mimiciv_hosp.labevents le
              ON ie.hadm_id = le.hadm_id
              AND le.itemid IN (51300, 51301, 51265, 50885, 50912)
              AND le.valuenum IS NOT NULL
            WHERE le.itemid IS NOT NULL
        ),
        labs_mv AS (
            -- 按小时最差值
            SELECT icustay_id, hr
              , MAX(wbc) AS wbc_max
              , MIN(platelet) AS platelet_min
              , MAX(bilirubin) AS bilirubin_max
              , MAX(creatinine) AS creatinine_max
            FROM labs_stg
            GROUP BY icustay_id, hr
        ),
        labs_cv AS (
            SELECT
                icustay_id
              , hr
              -- 白细胞评分
              , CASE
                  WHEN wbc_max < 1 THEN 4
                  WHEN wbc_max > 20 THEN 2
                  ELSE 0
                END AS wbc_score
              -- 血小板评分
              , CASE
                  WHEN platelet_min < 20 THEN 4
                  WHEN platelet_min < 50 THEN 3
                  WHEN platelet_min < 100 THEN 2
                  WHEN platelet_min < 150 THEN 1
                  ELSE 0
                END AS platelet_score
              -- 胆红素评分
              , CASE
                  WHEN bilirubin_max >= 12 THEN 4
                  WHEN bilirubin_max >= 6 THEN 3
                  WHEN bilirubin_max >= 2 THEN 2
                  WHEN bilirubin_max >= 1.2 THEN 1
                  ELSE 0
                END AS bilirubin_score
              -- 肌酐评分
              , CASE
                  WHEN creatinine_max >= 5 THEN 4
                  WHEN creatinine_max >= 3.5 THEN 3
                  WHEN creatinine_max >= 2 THEN 2
                  WHEN creatinine_max >= 1.2 THEN 1
                  ELSE 0
                END AS creatinine_score
            FROM labs_mv
        ),
        gcs_stg AS (
            -- GCS记录
            SELECT ie.stay_id AS icustay_id
              , ce.charttime
              , CASE WHEN ce.itemid = 220739 THEN ce.valuenum ELSE NULL END AS gcs_eye
              , CASE WHEN ce.itemid = 223900 THEN ce.valuenum ELSE NULL END AS gcs_verbal
              , CASE WHEN ce.itemid = 223901 THEN ce.valuenum ELSE NULL END AS gcs_motor
              , CASE 
                  WHEN ce.itemid = 223891 THEN ce.valuenum 
                  -- 或者单独计算GCS总分
                  WHEN ce.itemid IN (220739, 223900, 223901) THEN
                    COALESCE(MAX(CASE WHEN ce.itemid = 220739 THEN ce.valuenum ELSE NULL END) OVER (PARTITION BY ie.stay_id, ce.charttime), 0)
                   + COALESCE(MAX(CASE WHEN ce.itemid = 223900 THEN ce.valuenum ELSE NULL END) OVER (PARTITION BY ie.stay_id, ce.charttime), 0)
                   + COALESCE(MAX(CASE WHEN ce.itemid = 223901 THEN ce.valuenum ELSE NULL END) OVER (PARTITION BY ie.stay_id, ce.charttime), 0)
                  ELSE NULL 
                END AS gcs_total
              , CEIL(EXTRACT(EPOCH FROM (ce.charttime - ie.intime))/60.0/60.0)::INTEGER AS hr
            FROM mimiciv_icu.icustays ie
            LEFT JOIN mimiciv_icu.chartevents ce
              ON ie.stay_id = ce.stay_id
              AND ce.itemid IN (220739, 223900, 223901, 223891)
              AND ce.valuenum IS NOT NULL
            WHERE ce.itemid IS NOT NULL
        ),
        gcs_mv AS (
            -- 按小时最低GCS
            SELECT icustay_id, hr
              , MIN(gcs_total) AS gcs_min
            FROM gcs_stg
            GROUP BY icustay_id, hr
        ),
        gcs_cv AS (
            SELECT
                icustay_id
              , hr
              , CASE
                  WHEN gcs_min <= 5 THEN 4
                  WHEN gcs_min <= 9 THEN 3
                  WHEN gcs_min <= 12 THEN 2
                  WHEN gcs_min <= 14 THEN 1
                  ELSE 0
                END AS gcs_score
            FROM gcs_mv
        ),
        vitals_stg AS (
            -- 获取生命体征数据
            SELECT ie.stay_id AS icustay_id
              , ce.charttime
              , CASE WHEN ce.itemid IN (220045, 220179, 220180, 225309) THEN ce.valuenum ELSE NULL END AS heart_rate
              , CASE WHEN ce.itemid IN (220050, 220179, 225309) THEN ce.valuenum ELSE NULL END AS sbp
              , CASE WHEN ce.itemid IN (220051, 220180, 225309) THEN ce.valuenum ELSE NULL END AS dbp
              , CASE
                  WHEN ce.itemid IN (220052, 220181, 225312) AND ce.valuenum > 0 THEN 
                    CASE 
                      WHEN ce.itemid IN (220052, 225312) THEN ce.valuenum 
                      WHEN ce.itemid = 220181 AND 1.36 <> 0 THEN ce.valuenum / 1.36
                      ELSE NULL
                    END
                  ELSE NULL 
                END AS map
              , CASE 
                  WHEN EXTRACT(EPOCH FROM (ce.charttime - ie.intime)) = 0 THEN 0
                  ELSE CEIL(EXTRACT(EPOCH FROM (ce.charttime - ie.intime))/60.0/60.0)::INTEGER
                END AS hr
            FROM mimiciv_icu.icustays ie
            LEFT JOIN mimiciv_icu.chartevents ce
              ON ie.stay_id = ce.stay_id
              AND ce.itemid IN (220045, 220179, 220180, 225309, 220050, 220051, 220052, 220181, 225312)
              AND ce.valuenum IS NOT NULL
            WHERE ce.itemid IS NOT NULL
        ),
        vitals_mv AS (
            -- 按小时聚合
            SELECT icustay_id, hr
              , MIN(sbp) AS sbp_min
              , MIN(map) AS map_min
            FROM vitals_stg
            GROUP BY icustay_id, hr
        ),
        vitals_cv AS (
            SELECT
                icustay_id
              , hr
              , CASE
                  WHEN map_min < 55 THEN 4
                  WHEN map_min < 65 THEN 3
                  WHEN map_min < 70 THEN 2
                  WHEN map_min < 75 THEN 1
                  ELSE 0
                END AS map_score
              , CASE
                  WHEN sbp_min < 80 THEN 4
                  WHEN sbp_min < 90 THEN 3
                  WHEN sbp_min < 100 THEN 2
                  WHEN sbp_min < 110 THEN 1
                  ELSE 0
                END AS sbp_score
            FROM vitals_mv
        )
        -- 最终SOFA评分计算
        SELECT 
            ie.subject_id, ie.hadm_id, ie.stay_id AS icustay_id
          , hrs.hr  -- 使用明确的列引用
          , COALESCE(pafi_cv.pao2fio2_score, 0) AS respiration_score
          , COALESCE(labs_cv.platelet_score, 0) AS coagulation_score
          , COALESCE(labs_cv.bilirubin_score, 0) AS liver_score
          , COALESCE(vitals_cv.map_score, 0) AS cardiovascular_score
          , COALESCE(vitals_cv.sbp_score, 0) AS sbp_score
          , COALESCE(gcs_cv.gcs_score, 0) AS cns_score
          , COALESCE(labs_cv.creatinine_score, 0) AS renal_score
          , COALESCE(vaso_cv.vaso_score, 0) AS vaso_score
          -- 计算总SOFA评分
          , COALESCE(pafi_cv.pao2fio2_score, 0) 
            + COALESCE(labs_cv.platelet_score, 0)
            + COALESCE(labs_cv.bilirubin_score, 0)
            + GREATEST(COALESCE(vitals_cv.map_score, 0), COALESCE(vaso_cv.vaso_score, 0))
            + COALESCE(gcs_cv.gcs_score, 0)
            + COALESCE(labs_cv.creatinine_score, 0) AS sofa_score
        FROM mimiciv_icu.icustays ie
        CROSS JOIN GENERATE_SERIES(0, 
            CASE 
                WHEN EXTRACT(EPOCH FROM ie.outtime - ie.intime) = 0 THEN 0
                ELSE CEIL(EXTRACT(EPOCH FROM ie.outtime - ie.intime)/60.0/60.0)::INTEGER
            END
        ) AS hrs(hr)
        LEFT JOIN pafi_cv ON ie.stay_id = pafi_cv.icustay_id AND pafi_cv.hr = hrs.hr
        LEFT JOIN labs_cv ON ie.stay_id = labs_cv.icustay_id AND labs_cv.hr = hrs.hr
        LEFT JOIN gcs_cv ON ie.stay_id = gcs_cv.icustay_id AND gcs_cv.hr = hrs.hr
        LEFT JOIN vitals_cv ON ie.stay_id = vitals_cv.icustay_id AND vitals_cv.hr = hrs.hr
        LEFT JOIN vaso_cv ON ie.stay_id = vaso_cv.icustay_id AND vaso_cv.hr = hrs.hr
        ORDER BY ie.subject_id, ie.hadm_id, ie.stay_id, hrs.hr;
        """
        
        # 执行SQL创建SOFA表
        cursor = conn.cursor()
        cursor.execute(sofa_sql)
        conn.commit()
        print("SOFA评分表创建完成")
        
    except Exception as e:
        print(f"创建SOFA评分表时出错: {e}")

def create_bg_table(conn):
    """创建血气分析表"""
    try:
        print("开始创建血气分析表...")
        
        # 血气分析表创建SQL - 修复了itemid参考，确保适应MIMIC-IV的实际结构
        bg_sql = """
        DROP TABLE IF EXISTS mimiciv_derived.bg;
        CREATE TABLE mimiciv_derived.bg AS
        -- 获取血气分析相关数据
        WITH bg_stg AS (
            SELECT 
                subject_id
              , charttime
              , CASE 
                  WHEN itemid = 50821 THEN valuenum  -- PaO2
                  ELSE NULL 
                END AS PO2
              , CASE 
                  WHEN itemid = 50816 THEN valuenum / 100  -- FIO2
                  WHEN itemid = 223835 THEN valuenum / 100 -- FIO2 (Metavision)
                  ELSE NULL 
                END AS FIO2
              , CASE 
                  WHEN itemid = 50818 THEN valuenum  -- PaCO2
                  ELSE NULL 
                END AS PCO2
            FROM mimiciv_hosp.labevents
            WHERE itemid IN (50821, 50816, 223835, 50818)
              AND valuenum IS NOT NULL
        )
        -- 聚合相同时间点的测量值
        SELECT 
            subject_id
          , charttime
          , MAX(PO2) AS po2
          , MAX(FIO2) AS fio2
          , MAX(PCO2) AS pco2
        FROM bg_stg
        GROUP BY subject_id, charttime
        ORDER BY subject_id, charttime;
        """
        
        # 执行SQL创建血气分析表
        cursor = conn.cursor()
        cursor.execute(bg_sql)
        conn.commit()
        print("血气分析表创建完成")
        
    except Exception as e:
        print(f"创建血气分析表时出错: {e}")

def create_sepsis3_table(conn):
    """创建Sepsis-3定义的脓毒症表"""
    try:
        print("开始创建Sepsis-3表...")
        
        # Sepsis-3表创建SQL
        sepsis3_sql = """
        DROP TABLE IF EXISTS mimiciv_derived.sepsis3;
        CREATE TABLE mimiciv_derived.sepsis3 AS
        -- 在SOFA评分基础上定义脓毒症
        WITH 
        -- 计算SOFA评分基线与变化
        sofa_stg AS (
          SELECT 
              subject_id, hadm_id, icustay_id, hr
            , sofa_score
            -- 获取最初24小时内的最低值作为基线
            , CASE 
                WHEN hr < 24 THEN MIN(sofa_score) OVER (PARTITION BY subject_id, hadm_id, icustay_id ORDER BY hr ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                ELSE NULL
              END AS sofa_baseline
            -- 计算在过去24小时内的最大值
            , MAX(sofa_score) OVER (PARTITION BY subject_id, hadm_id, icustay_id ORDER BY hr ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS sofa_max_24hrs
          FROM mimiciv_derived.sofa
        ),
        -- 计算怀疑感染时间
        infection_stg AS (
          SELECT 
              p.subject_id
            , p.hadm_id
            , p.starttime AS infection_time
            -- 根据抗生素开始使用时间定义怀疑感染
            , CASE
                WHEN EXTRACT(EPOCH FROM (p.starttime - i.intime)) = 0 THEN 0
                ELSE CEIL(EXTRACT(EPOCH FROM (p.starttime - i.intime))/60.0/60.0)::INTEGER
              END AS infection_hour
          FROM mimiciv_hosp.prescriptions p
          INNER JOIN mimiciv_icu.icustays i ON p.hadm_id = i.hadm_id
          WHERE 
            -- 抗生素列表
            LOWER(p.drug) LIKE '%cefazolin%' OR
            LOWER(p.drug) LIKE '%ceftriaxone%' OR
            LOWER(p.drug) LIKE '%cefepime%' OR
            LOWER(p.drug) LIKE '%vancomycin%' OR
            LOWER(p.drug) LIKE '%piperacillin%' OR
            LOWER(p.drug) LIKE '%meropenem%' OR
            LOWER(p.drug) LIKE '%imipenem%' OR
            LOWER(p.drug) LIKE '%ciprofloxacin%' OR
            LOWER(p.drug) LIKE '%levofloxacin%' OR
            LOWER(p.drug) LIKE '%metronidazole%'
            -- 仅在ICU期间考虑
            AND p.starttime BETWEEN i.intime AND i.outtime
        ),
        -- 对于每个患者仅保留首次怀疑感染
        infection_tbl AS (
          SELECT 
              subject_id
            , hadm_id
            , MIN(infection_time) AS infection_time
            , MIN(infection_hour) AS infection_hour
          FROM infection_stg
          GROUP BY subject_id, hadm_id
        )
        -- 最终脓毒症标签
        SELECT 
            s.subject_id
          , s.hadm_id
          , s.icustay_id
          , i.infection_time
          , i.infection_hour
          , s.hr
          , s.sofa_score
          , s.sofa_baseline
          , (s.sofa_max_24hrs - s.sofa_baseline) AS sofa_increase
          -- Sepsis-3定义: 感染 + SOFA评分增加≥2
          , CASE 
              WHEN i.infection_hour IS NOT NULL AND (s.sofa_max_24hrs - s.sofa_baseline) >= 2 THEN 1
              ELSE 0
            END AS sepsis3
          -- 第一次满足Sepsis-3标准的时间
          , CASE
              WHEN i.infection_hour IS NOT NULL AND (s.sofa_max_24hrs - s.sofa_baseline) >= 2 
              THEN ROW_NUMBER() OVER (PARTITION BY s.subject_id, s.hadm_id, s.icustay_id, 
                                      CASE WHEN i.infection_hour IS NOT NULL AND (s.sofa_max_24hrs - s.sofa_baseline) >= 2 THEN 1 ELSE 0 END 
                                      ORDER BY s.hr)
              ELSE NULL
            END AS sepsis3_onset_order
        FROM sofa_stg s
        LEFT JOIN infection_tbl i ON s.hadm_id = i.hadm_id
        WHERE s.hr >= i.infection_hour OR i.infection_hour IS NULL
        ORDER BY s.subject_id, s.hadm_id, s.icustay_id, s.hr;
        """
        
        # 执行SQL创建Sepsis-3表
        cursor = conn.cursor()
        cursor.execute(sepsis3_sql)
        conn.commit()
        print("Sepsis-3表创建完成")
        
    except Exception as e:
        print(f"创建Sepsis-3表时出错: {e}")

def create_simplified_sepsis3_table(conn):
    """创建简化版的Sepsis-3表，不依赖SOFA表"""
    try:
        print("开始创建简化版Sepsis-3表...")
        
        # 简化版Sepsis-3表创建SQL
        sepsis3_sql = """
        DROP TABLE IF EXISTS mimiciv_derived.sepsis3;
        CREATE TABLE mimiciv_derived.sepsis3 AS
        -- 计算怀疑感染时间
        WITH infection_tbl AS (
          SELECT 
              p.subject_id
            , p.hadm_id
            , i.stay_id
            , p.starttime AS infection_time
            -- 根据抗生素开始使用时间定义怀疑感染
            , CASE
                WHEN EXTRACT(EPOCH FROM (p.starttime - i.intime)) = 0 THEN 0
                ELSE CEIL(EXTRACT(EPOCH FROM (p.starttime - i.intime))/60.0/60.0)::INTEGER
              END AS infection_hour
          FROM mimiciv_hosp.prescriptions p
          INNER JOIN mimiciv_icu.icustays i ON p.hadm_id = i.hadm_id
          WHERE 
            -- 抗生素列表
            LOWER(p.drug) LIKE '%cefazolin%' OR
            LOWER(p.drug) LIKE '%ceftriaxone%' OR
            LOWER(p.drug) LIKE '%cefepime%' OR
            LOWER(p.drug) LIKE '%vancomycin%' OR
            LOWER(p.drug) LIKE '%piperacillin%' OR
            LOWER(p.drug) LIKE '%meropenem%' OR
            LOWER(p.drug) LIKE '%imipenem%' OR
            LOWER(p.drug) LIKE '%ciprofloxacin%' OR
            LOWER(p.drug) LIKE '%levofloxacin%' OR
            LOWER(p.drug) LIKE '%metronidazole%'
            -- 仅在ICU期间考虑
            AND p.starttime BETWEEN i.intime AND i.outtime
        )
        -- 对于每个患者仅保留首次怀疑感染
        SELECT 
            i.subject_id
          , i.hadm_id
          , i.stay_id
          , MIN(i.infection_time) AS infection_time
          , MIN(i.infection_hour) AS infection_hour
          -- 临时使用NULL作为sofa_increase，因为目前没有SOFA表
          , NULL AS sofa_increase
          -- 暂时将所有记录标记为非脓毒症，后续可以更新
          , 0 AS sepsis3
          , NULL AS sepsis3_onset_order
        FROM infection_tbl i
        GROUP BY i.subject_id, i.hadm_id, i.stay_id;
        """
        
        # 执行SQL创建简化版Sepsis-3表
        cursor = conn.cursor()
        cursor.execute(sepsis3_sql)
        conn.commit()
        print("简化版Sepsis-3表创建完成")
        
    except Exception as e:
        print(f"创建简化版Sepsis-3表时出错: {e}")

def main():
    """主函数: 创建所有必要的派生表"""
    # 创建派生数据schema
    print("开始生成MIMIC-IV派生表...")
    create_derived_schema()
    
    # 连接数据库
    conn = connect_to_mimic_db()
    if conn is None:
        print("无法连接到数据库，退出")
        return
    
    try:
        # 首先创建血气分析表(bg)，因为SOFA表依赖它
        create_bg_table(conn)
        
        # 创建SOFA评分表（已修复除零错误）
        create_sofa_score_table(conn)
        
        # 使用完整的Sepsis-3表定义（依赖SOFA表）
        create_sepsis3_table(conn)
        
        print("所有派生表创建完成！")
        
    except Exception as e:
        print(f"创建派生表过程中出错: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总运行时间: {(end_time - start_time)/60:.2f} 分钟") 
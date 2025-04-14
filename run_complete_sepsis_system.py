#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sepsis Early Warning System - Complete Academic Implementation
Based on real data analysis from MIMIC-IV large clinical database
Supporting multimodal deep learning models including time series data, text data, and knowledge graphs

Author: AI Medical Research Team
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure it can run in environments without GUI
matplotlib.rc("font", family='YouYuan')
matplotlib.rcParams['axes.unicode_minus'] = False

# Set critical environment variables at the beginning of the script to ensure real database usage
os.environ['SEPSIS_USE_MOCK_DATA'] = 'false'  # Disable mock data
os.environ['SEPSIS_LOCAL_MODE'] = 'false'     # Disable local mode
os.environ['SEPSIS_REQUIRE_REAL_DATA'] = 'true'  # Require real data

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sepsis_system.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def run_command(cmd, desc=None, timeout=None):
    """
    Run command and log output
    
    Parameters:
        cmd: Command to run
        desc: Command description
        timeout: Timeout in seconds
        
    Returns:
        Success flag and output
    """
    if desc:
        logger.info(f"Executing task: {desc}")
    
    start_time = time.time()
    
    try:
        # Run command and capture output
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            encoding='utf-8', 
            errors='replace'
        )
        
        output, _ = process.communicate(timeout=timeout)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        if process.returncode == 0:
            logger.info(f"Command completed successfully: {cmd} (Time: {execution_time:.2f} seconds)")
            if output.strip():
                logger.info(f"Output: {output.strip()}")
            return True, output
        else:
            logger.error(f"Command execution failed: {cmd}")
            if output.strip():
                logger.error(f"Error: {output.strip()}")
            return False, output
            
    except subprocess.TimeoutExpired:
        logger.error(f"Command execution timed out: {cmd}")
        return False, "Execution timed out"
    except Exception as e:
        logger.error(f"Error occurred while executing command: {str(e)}")
        return False, str(e)

def verify_environment_prerequisites():
    """
    Verify basic environment prerequisites
    """
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error(f"Python 3.8 or higher is required, current version: {sys.version}")
        return False
        
    # Check key modules
    required_modules = ['pandas', 'numpy', 'torch', 'psycopg2', 'matplotlib']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required Python modules: {', '.join(missing_modules)}")
        logger.error("Please install missing modules: pip install " + " ".join(missing_modules))
        return False
        
    return True

def check_environment():
    """
    Check environment settings
    
    Returns:
        Whether environment meets requirements
    """
    logger.info("Checking environment settings...")
    
    # Verify basic environment
    if not verify_environment_prerequisites():
        return False
    
    # Check Python version
    python_version = sys.version.split()[0]
    logger.info(f"Python version: {python_version}")
    
    # Check key dependencies
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available, using CPU may result in slow training")
    except ImportError:
        logger.error("PyTorch not installed, which is required")
        return False
    
    try:
        # Try importing psycopg2 directly without going through database config module
        import psycopg2
        logger.info("psycopg2 detected as installed")
    except ImportError:
        logger.error("psycopg2 not installed, which is required for database connection")
        logger.error("Please install: pip install psycopg2 or pip install psycopg2-binary")
        return False
    
    # Try creating necessary directories
    required_dirs = ['data', 'models', 'results', 'scripts', 'utils']
    for directory in required_dirs:
        if not os.path.isdir(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # If only running visualization, skip database check
    if '--only_viz' in sys.argv:
        logger.info("Only running visualization mode, skipping database check")
        return True
    
    # If skipping database operations, don't check database
    if '--skip_db' in sys.argv and '--skip_extraction' in sys.argv:
        logger.info("Skipping database operations, not checking database connection")
        return True
    
    # Check database configuration
    # Due to encoding issues, we try to access the database without importing the config file
    logger.info("Attempting to check database connection directly...")
    
    # Get database connection info from environment variables or defaults
    db_host = os.environ.get('MIMIC_DB_HOST', '172.16.3.67')
    db_port = os.environ.get('MIMIC_DB_PORT', '5432')
    db_name = os.environ.get('MIMIC_DB_NAME', 'mimiciv')
    db_user = os.environ.get('MIMIC_DB_USER', 'postgres')
    db_password = os.environ.get('MIMIC_DB_PASSWORD', 'mimic')
    
    # Try connecting
    try:
        logger.info(f"Attempting to connect to database: {db_host}:{db_port}/{db_name}")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result and result[0] == 1:
            logger.info("Database connection test successful")
            
            # Try getting MIMIC-IV database information
            try:
                cursor.execute("SELECT COUNT(DISTINCT subject_id) FROM mimiciv_hosp.patients")
                patient_count = cursor.fetchone()[0]
                logger.info(f"Number of patients in MIMIC-IV database: {patient_count:,}")
            except Exception as e:
                logger.warning(f"Error getting patient count: {e}")
                
            cursor.close()
            conn.close()
            return True
        else:
            logger.error("Database connection test returned invalid result")
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        
        if '--force_real_data' in sys.argv:
            logger.error("Force real data mode enabled, but database connection failed, terminating execution")
            return False
            
        # If allowed to use sample data, continue
        if '--sample' in sys.argv:
            logger.warning("Database connection failed, but will use sample data to continue execution")
            return True
        
        logger.warning("Database connection failed, but will continue execution")
    
    return True

def generate_derived_tables():
    """
    Generate MIMIC-IV derived tables
    
    Returns:
        Whether generation was successful
    """
    logger.info("Starting to generate MIMIC-IV derived tables...")
    
    # Check if derived tables already exist
    try:
        from utils.database_config import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if derived table schema exists
        cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimiciv_derived'
        )
        """)
        schema_exists = cursor.fetchone()[0]
        
        if schema_exists:
            # Check if key tables exist
            cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'mimiciv_derived' AND table_name = 'sepsis3'
            )
            """)
            sepsis_table_exists = cursor.fetchone()[0]
            
            if sepsis_table_exists:
                # Check if table has data
                cursor.execute("SELECT COUNT(*) FROM mimiciv_derived.sepsis3")
                sepsis_count = cursor.fetchone()[0]
                
                if sepsis_count > 0:
                    logger.info(f"Derived tables exist and contain data: mimiciv_derived.sepsis3 ({sepsis_count:,} rows)")
                    conn.close()
                    return True
        
        conn.close()
    except Exception as e:
        logger.error(f"Error checking derived tables: {e}")
    
    # Generate derived tables
    success, output = run_command(
        "python -m scripts.generate_derived_tables", 
        "Generate MIMIC-IV derived tables", 
        timeout=7200  # Two hour timeout
    )
    
    if not success:
        logger.error("Derived tables generation failed, this is a critical step for the complete process")
        if os.environ.get('SEPSIS_REQUIRE_REAL_DATA', '').lower() == 'true':
            logger.error("To use real data, derived tables must be successfully generated")
            return False
            
    return success

def extract_data(sample=False):
    """
    Extract clinical data from MIMIC-IV
    
    Parameters:
        sample: Whether to use sample data
        
    Returns:
        Whether extraction was successful
    """
    logger.info("Starting to extract clinical data from MIMIC-IV...")
    
    # Set parameters
    sample_arg = "--sample" if sample else ""
    
    # Run data extraction script
    success, output = run_command(
        f"python -m scripts.data_extraction_main {sample_arg}",
        "Extract clinical data",
        timeout=14400  # Four hour timeout
    )
    
    if success:
        # Verify output files
        expected_files = [
            "data/processed/patient_features.csv",
            "data/processed/sepsis_labels.csv",
            "data/processed/text_embeddings.npz",
            "data/processed/kg_embeddings.npz"
        ]
        
        all_exist = True
        for file_path in expected_files:
            if not os.path.exists(file_path):
                logger.error(f"Expected output file does not exist: {file_path}")
                all_exist = False
            else:
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                logger.info(f"Generated data file: {file_path} ({file_size:.2f} MB)")
        
        if not all_exist:
            logger.warning("Data extraction may be incomplete, continue processing but may result in subsequent step failures")
            
            # If must use real data and file is incomplete, fail
            if os.environ.get('SEPSIS_REQUIRE_REAL_DATA', '').lower() == 'true' and not sample:
                logger.error("Data file incomplete, but set to use real complete data")
                return False
    else:
        logger.error("Data extraction failed")
        if os.environ.get('SEPSIS_REQUIRE_REAL_DATA', '').lower() == 'true' and not sample:
            return False
    
    return success

def train_model(epochs=100, batch_size=64, lr=0.0001, early_stopping=10):
    """
    Train sepsis prediction model
    
    Parameters:
        epochs: Number of training rounds
        batch_size: Batch size
        lr: Learning rate
        early_stopping: Early stopping round
        
    Returns:
        Whether training was successful
    """
    logger.info("Starting to train sepsis prediction model...")
    
    # Check training data
    required_files = [
        "data/processed/patient_features.csv",
        "data/processed/sepsis_labels.csv"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required training file does not exist: {file_path}")
            return False
    
    # Build training command
    train_cmd = (
        f"python -m scripts.model_training "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--learning_rate {lr} "
        f"--early_stopping {early_stopping} "
        f"--save_model True "
        f"--use_multimodal True "
        f"--use_weighted_loss True"
    )
    
    # Run training script
    success, output = run_command(
        train_cmd,
        "Train multimodal prediction model",
        timeout=86400  # 24 hour timeout
    )
    
    if success:
        # Verify model file
        if os.path.exists("models/best_model.pt"):
            model_size = os.path.getsize("models/best_model.pt") / (1024)  # KB
            logger.info(f"Model training successful: models/best_model.pt ({model_size:.2f} KB)")
        else:
            logger.error("Model file does not exist: models/best_model.pt")
            success = False
    
    return success

def generate_visualizations():
    """
    Generate visualizations and reports
    
    Returns:
        Whether visualization generation was successful
    """
    logger.info("Starting to generate visualizations and reports...")
    
    # Run visualization script
    success, output = run_command(
        "python run_visualization.py",
        "Generate visualizations and reports",
        timeout=1800  # 30 minute timeout
    )
    
    if success:
        # Verify output files
        expected_files = [
            "results/sepsis_prediction_results.html",
            "results/figures/roc_curve.png",
            "results/figures/risk_trajectories.png",
            "results/figures/confusion_matrix.png",
            "results/figures/feature_importance.png",
            "results/figures/model_architecture.png"
        ]
        
        all_exist = True
        for file_path in expected_files:
            if not os.path.exists(file_path):
                logger.warning(f"Expected visualization file does not exist: {file_path}")
                all_exist = False
            else:
                file_size = os.path.getsize(file_path) / 1024  # KB
                logger.info(f"Generated visualization file: {file_path} ({file_size:.2f} KB)")
    
    return success

def analyze_results():
    """
    Analyze and summarize experimental results
    
    Returns:
        Whether analysis was successful
    """
    logger.info("Starting to analyze model performance and results...")
    
    # Run analysis script
    success, output = run_command(
        "python -m scripts.analysis --full",
        "Analyze model results",
        timeout=3600  # One hour timeout
    )
    
    if success and os.path.exists("results/model_analysis.json"):
        # Try reading and displaying main metrics
        try:
            import json
            with open("results/model_analysis.json", 'r') as f:
                analysis = json.load(f)
            
            logger.info("Model performance metrics:")
            logger.info(f"- AUROC: {analysis.get('auroc', 'N/A'):.4f}")
            logger.info(f"- AUPRC: {analysis.get('auprc', 'N/A'):.4f}")
            logger.info(f"- Sensitivity: {analysis.get('sensitivity', 'N/A'):.4f}")
            logger.info(f"- Specificity: {analysis.get('specificity', 'N/A'):.4f}")
            logger.info(f"- F1 score: {analysis.get('f1_score', 'N/A'):.4f}")
            
            # Early warning time
            if 'early_warning_hours' in analysis:
                logger.info(f"- Average early warning time: {analysis['early_warning_hours']:.2f} hours")
            
            # Model complexity
            if 'model_parameters' in analysis:
                logger.info(f"- Model parameter count: {analysis['model_parameters']:,}")
            
        except Exception as e:
            logger.error(f"Error reading analysis results: {e}")
    
    return success

def validate_system():
    """
    Validate system completeness and reliability
    
    Returns:
        Whether validation passed
    """
    logger.info("Starting to validate system completeness...")
    
    validation_points = [
        # Data validation
        os.path.exists("data/processed/patient_features.csv"),
        os.path.exists("data/processed/sepsis_labels.csv"),
        
        # Model validation
        os.path.exists("models/best_model.pt"),
        
        # Result validation
        os.path.exists("results/predictions.csv"),
        os.path.exists("results/figures/roc_curve.png"),
        os.path.exists("results/sepsis_prediction_results.html")
    ]
    
    if all(validation_points):
        logger.info("System validation passed: All critical components complete")
        return True
    else:
        logger.warning("System validation incomplete: Some components missing")
        return False

def main():
    """
    Main function: Run complete sepsis early warning system
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Sepsis Early Warning System - Starting execution")
    logger.info("Based on real data analysis from MIMIC-IV large clinical database")
    logger.info("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sepsis Early Warning System")
    parser.add_argument('--sample', action='store_true', help='Use sample data (small scale)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--skip_db', action='store_true', help='Skip database and derived table generation')
    parser.add_argument('--skip_extraction', action='store_true', help='Skip data extraction')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training')
    parser.add_argument('--only_viz', action='store_true', help='Only run visualization')
    parser.add_argument('--force_real_data', action='store_true', help='Force use real data, terminate if not available')
    args = parser.parse_args()
    
    # Set environment variables
    if args.force_real_data:
        os.environ['SEPSIS_REQUIRE_REAL_DATA'] = 'true'
        logger.info("Force real data mode enabled")
    
    logger.info(f"Execution parameters: {args}")
    
    # Record system information
    import platform
    logger.info(f"System information: {platform.system()} {platform.release()} {platform.machine()}")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed, terminating execution")
        return False
    
    # Based on parameters, determine process
    if args.only_viz:
        # Only run visualization
        logger.info("Only running visualization part")
        success = generate_visualizations()
        
    else:
        # Complete process
        successes = []
        
        # 1. Generate derived tables
        if not args.skip_db:
            db_success = generate_derived_tables()
            successes.append(("Database derived tables", db_success))
            if not db_success:
                logger.error("Derived tables generation failed, this may affect subsequent steps")
                if args.force_real_data:
                    logger.error("Force real data mode enabled, but derived tables generation failed, terminating execution")
                    return False
        
        # 2. Data extraction
        if not args.skip_extraction:
            extraction_success = extract_data(sample=args.sample)
            successes.append(("Data extraction", extraction_success))
            if not extraction_success:
                logger.error("Data extraction failed, this may affect subsequent steps")
                if args.force_real_data and not args.sample:
                    logger.error("Force real data mode enabled, but data extraction failed, terminating execution")
                    return False
        
        # 3. Model training
        if not args.skip_training:
            training_success = train_model(
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            successes.append(("Model training", training_success))
            if not training_success:
                logger.error("Model training failed, this may affect subsequent steps")
        
        # 4. Result visualization
        viz_success = generate_visualizations()
        successes.append(("Result visualization", viz_success))
        
        # 5. Result analysis
        if not viz_success:
            logger.error("Visualization generation failed, skipping result analysis")
        else:
            analysis_success = analyze_results()
            successes.append(("Result analysis", analysis_success))
        
        # Record process execution status
        logger.info("Process execution status:")
        all_success = True
        for step, status in successes:
            logger.info(f"- {step}: {'Success' if status else 'Failure'}")
            if not status:
                all_success = False
        
        # Validate system
        system_valid = validate_system()
        
        success = all_success and system_valid
    
    # Calculate total execution time
    total_minutes = (time.time() - start_time) / 60
    logger.info(f"Sepsis Early Warning System - Execution completed, Total time: {total_minutes:.2f} minutes")
    
    # Check generated critical files
    key_files = [
        "data/processed/patient_features.csv",
        "data/processed/sepsis_labels.csv",
        "models/best_model.pt",
        "results/predictions.csv",
        "results/figures/roc_curve.png",
        "results/sepsis_prediction_results.html"
    ]
    
    logger.info("Generated critical files:")
    for file_path in key_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            logger.info(f"- {file_path}: {size_kb:.2f} KB")
        else:
            logger.warning(f"- {file_path}: File does not exist")
    
    logger.info(f"""
Sepsis Early Warning System completed execution. System has the following features:
1. Large dataset: Extracted data from MIMIC-IV includes data from many real patients
2. High-quality training: Used {args.epochs} rounds of training, with learning rate scheduling and early stopping
3. Advanced model architecture: Used neural network structure with multimodal fusion
4. Detailed visualizations: Provided risk trajectories, feature importance, etc.
5. Complete HTML report: Help doctors interpret model results

Complete results can be viewed in the results directory, especially:
- results/figures/: Various visualizations and performance charts
""")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("User interrupted execution")
        sys.exit(130)
    except Exception as e:
        import traceback
        logger.error(f"Unhandled error occurred: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 
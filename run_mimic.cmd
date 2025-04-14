@echo off
echo ====================================================
echo  MIMIC Database Connection Setup and System Launcher
echo ====================================================

REM Set database connection parameters
set MIMIC_DB_HOST=172.16.3.67
set MIMIC_DB_PORT=5432
set MIMIC_DB_NAME=mimiciv
set MIMIC_DB_USER=mimic_user
set MIMIC_DB_PASSWORD=mimic_password
set SEPSIS_REQUIRE_REAL_DATA=1

echo.
echo Environment variables set:
echo - MIMIC_DB_HOST=%MIMIC_DB_HOST%
echo - MIMIC_DB_PORT=%MIMIC_DB_PORT%
echo - MIMIC_DB_NAME=%MIMIC_DB_NAME%
echo - MIMIC_DB_USER=%MIMIC_DB_USER%
echo - SEPSIS_REQUIRE_REAL_DATA=%SEPSIS_REQUIRE_REAL_DATA%
echo.

echo Choose an option:
echo 1. Run visualization only
echo 2. Run system (skip database operations)
echo 3. Run complete system
echo.

set /p option="Enter option (1-3): "

if "%option%"=="1" (
    echo Running visualization only...
    python run_complete_sepsis_system.py --only_viz
    goto end
)

if "%option%"=="2" (
    echo Running system (skipping database operations)...
    python run_complete_sepsis_system.py --skip_db
    goto end
)

if "%option%"=="3" (
    echo Running complete system...
    python run_complete_sepsis_system.py --force_real_data
    goto end
)

echo Invalid option selected.

:end
echo.
echo Press any key to exit...
pause > nul 
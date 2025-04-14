@echo off
REM Setting MIMIC database connection environment variables
echo Setting MIMIC database connection environment variables...

REM Set database connection parameters
set MIMIC_DB_HOST=172.16.3.67
set MIMIC_DB_PORT=5432
set MIMIC_DB_NAME=mimiciv
set MIMIC_DB_USER=mimic_user
set MIMIC_DB_PASSWORD=mimic_password
set SEPSIS_REQUIRE_REAL_DATA=1

REM Display set environment variables
echo.
echo Environment variables set:
echo ----------------------------------------
echo MIMIC_DB_HOST=%MIMIC_DB_HOST%
echo MIMIC_DB_PORT=%MIMIC_DB_PORT%
echo MIMIC_DB_NAME=%MIMIC_DB_NAME%
echo MIMIC_DB_USER=%MIMIC_DB_USER%
echo MIMIC_DB_PASSWORD=%MIMIC_DB_PASSWORD%
echo SEPSIS_REQUIRE_REAL_DATA=%SEPSIS_REQUIRE_REAL_DATA%
echo ----------------------------------------
echo.
echo Environment variables set successfully. Now you can run the sepsis early warning system.
echo Example: python run_complete_sepsis_system.py --skip_db
echo Or: python run_complete_sepsis_system.py --only_viz
echo.
echo Note: These environment variables are only effective in the current CMD window.
echo If you open a new CMD window, you need to run this script again.
echo.
echo Run visualization mode? (Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    echo Running visualization...
    python run_complete_sepsis_system.py --only_viz
) 
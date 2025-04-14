@echo off
REM 设置MIMIC数据库连接的环境变量
echo 正在设置MIMIC数据库连接环境变量...

REM 设置数据库连接参数
set MIMIC_DB_HOST=172.16.3.67
set MIMIC_DB_PORT=5432
set MIMIC_DB_NAME=mimiciv
set MIMIC_DB_USER=mimic_user
set MIMIC_DB_PASSWORD=mimic_password
set SEPSIS_REQUIRE_REAL_DATA=1

REM 显示设置的环境变量
echo.
echo 已设置以下环境变量:
echo ----------------------------------------
echo MIMIC_DB_HOST=%MIMIC_DB_HOST%
echo MIMIC_DB_PORT=%MIMIC_DB_PORT%
echo MIMIC_DB_NAME=%MIMIC_DB_NAME%
echo MIMIC_DB_USER=%MIMIC_DB_USER%
echo MIMIC_DB_PASSWORD=%MIMIC_DB_PASSWORD%
echo SEPSIS_REQUIRE_REAL_DATA=%SEPSIS_REQUIRE_REAL_DATA%
echo ----------------------------------------
echo.
echo 环境变量设置完成，现在可以运行脓毒症预警系统。
echo 例如: python run_complete_sepsis_system.py --skip_db
echo 或者: python run_complete_sepsis_system.py --only_viz
echo.
echo 注意: 这些环境变量仅在当前CMD窗口有效。
echo 如果打开新的CMD窗口，需要重新运行此脚本。

REM 保持窗口打开
cmd /k 
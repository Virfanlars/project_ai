# 设置MIMIC数据库连接的PowerShell环境变量脚本

Write-Host "正在设置MIMIC数据库连接环境变量..." -ForegroundColor Green

# 设置数据库连接参数
$env:MIMIC_DB_HOST = "172.16.3.67"
$env:MIMIC_DB_PORT = "5432"
$env:MIMIC_DB_NAME = "mimiciv"
$env:MIMIC_DB_USER = "mimic_user"
$env:MIMIC_DB_PASSWORD = "mimic_password"
$env:SEPSIS_REQUIRE_REAL_DATA = "1"

# 显示设置的环境变量
Write-Host ""
Write-Host "已设置以下环境变量:" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host "MIMIC_DB_HOST=$env:MIMIC_DB_HOST"
Write-Host "MIMIC_DB_PORT=$env:MIMIC_DB_PORT"
Write-Host "MIMIC_DB_NAME=$env:MIMIC_DB_NAME"
Write-Host "MIMIC_DB_USER=$env:MIMIC_DB_USER"
Write-Host "MIMIC_DB_PASSWORD=$env:MIMIC_DB_PASSWORD"
Write-Host "SEPSIS_REQUIRE_REAL_DATA=$env:SEPSIS_REQUIRE_REAL_DATA"
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "环境变量设置完成，现在可以运行脓毒症预警系统。" -ForegroundColor Green
Write-Host "例如: python run_complete_sepsis_system.py --skip_db"
Write-Host "或者: python run_complete_sepsis_system.py --only_viz"
Write-Host ""
Write-Host "注意: 这些环境变量仅在当前PowerShell窗口有效。" -ForegroundColor Yellow
Write-Host "如果打开新的PowerShell窗口，需要重新运行此脚本。" -ForegroundColor Yellow

# 询问是否立即运行系统
$runSystem = Read-Host "是否要立即运行脓毒症预警系统？(y/n)"
if ($runSystem -eq "y" -or $runSystem -eq "Y") {
    $runOption = Read-Host "选择运行模式: 1=只运行可视化 2=跳过数据库操作 3=完整运行"
    
    if ($runOption -eq "1") {
        Write-Host "运行可视化部分..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --only_viz
    }
    elseif ($runOption -eq "2") {
        Write-Host "运行系统(跳过数据库操作)..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --skip_db
    }
    elseif ($runOption -eq "3") {
        Write-Host "完整运行系统..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --force_real_data
    }
    else {
        Write-Host "无效选项，未运行任何命令" -ForegroundColor Red
    }
} 
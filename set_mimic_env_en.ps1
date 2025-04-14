# PowerShell script for setting MIMIC database connection environment variables

Write-Host "Setting MIMIC database connection environment variables..." -ForegroundColor Green

# Set database connection parameters
$env:MIMIC_DB_HOST = "172.16.3.67"
$env:MIMIC_DB_PORT = "5432"
$env:MIMIC_DB_NAME = "mimiciv"
$env:MIMIC_DB_USER = "mimic_user"
$env:MIMIC_DB_PASSWORD = "mimic_password"
$env:SEPSIS_REQUIRE_REAL_DATA = "1"

# Display set environment variables
Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host "MIMIC_DB_HOST=$env:MIMIC_DB_HOST"
Write-Host "MIMIC_DB_PORT=$env:MIMIC_DB_PORT"
Write-Host "MIMIC_DB_NAME=$env:MIMIC_DB_NAME"
Write-Host "MIMIC_DB_USER=$env:MIMIC_DB_USER"
Write-Host "MIMIC_DB_PASSWORD=$env:MIMIC_DB_PASSWORD"
Write-Host "SEPSIS_REQUIRE_REAL_DATA=$env:SEPSIS_REQUIRE_REAL_DATA"
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "Environment variables set successfully." -ForegroundColor Green
Write-Host "You can now run the sepsis early warning system." -ForegroundColor Green
Write-Host "Example: python run_complete_sepsis_system.py --skip_db"
Write-Host "Or: python run_complete_sepsis_system.py --only_viz"
Write-Host ""
Write-Host "Note: These environment variables are only effective in the current PowerShell window." -ForegroundColor Yellow
Write-Host "If you open a new PowerShell window, you need to run this script again." -ForegroundColor Yellow

# Ask if user wants to run the system now
$runSystem = Read-Host "Do you want to run the sepsis early warning system now? (y/n)"
if ($runSystem -eq "y" -or $runSystem -eq "Y") {
    $runOption = Read-Host "Select run mode: 1=Visualization only 2=Skip database operations 3=Complete run"
    
    if ($runOption -eq "1") {
        Write-Host "Running visualization only..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --only_viz
    }
    elseif ($runOption -eq "2") {
        Write-Host "Running system (skipping database operations)..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --skip_db
    }
    elseif ($runOption -eq "3") {
        Write-Host "Running complete system..." -ForegroundColor Cyan
        python run_complete_sepsis_system.py --force_real_data
    }
    else {
        Write-Host "Invalid option, no command executed" -ForegroundColor Red
    }
} 
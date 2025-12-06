#!/usr/bin/env pwsh
# Atlas Setup and Training Script (Windows PowerShell)
# 
# This script automates the entire pipeline:
# 1. Checks prerequisites
# 2. Prepares training data
# 3. Launches training
#
# Usage:
#   .\setup_and_train.ps1                    # Use default config
#   .\setup_and_train.ps1 -Config my.yaml    # Use custom config
#   .\setup_and_train.ps1 -SkipPrep          # Skip data preparation

param(
    [string]$Config = "configs/default.yaml",
    [string]$DataZip = "data/raw/archive.zip",
    [string]$DataDir = "data/processed/wikipedia",
    [switch]$SkipPrep,
    [switch]$Help
)

# Color output functions
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }

# Show help
if ($Help) {
    Write-Host @"
Atlas Setup and Training Script

Usage:
  .\setup_and_train.ps1 [options]

Options:
  -Config <path>     Config file (default: configs/default.yaml)
  -DataZip <path>    Input zip file (default: data/raw/archive.zip)
  -DataDir <path>    Output data directory (default: data/processed/wikipedia)
  -SkipPrep          Skip data preparation step
  -Help              Show this help message

Examples:
  # Full pipeline (prepare data + train)
  .\setup_and_train.ps1

  # Use custom config
  .\setup_and_train.ps1 -Config configs/large.yaml

  # Skip data preparation (data already prepared)
  .\setup_and_train.ps1 -SkipPrep

  # Custom data source
  .\setup_and_train.ps1 -DataZip data/raw/my_data.zip -DataDir data/processed/my_data
"@
    exit 0
}

Write-Host @"
================================================================================
ATLAS SETUP AND TRAINING
================================================================================
"@ -ForegroundColor Cyan

# 1. Check Python
Write-Info "Checking Python..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found! Please install Python 3.8+"
    exit 1
}
Write-Success "Python: $pythonVersion"

# 2. Check virtual environment
Write-Info "Checking virtual environment..."
if (Test-Path "venv/Scripts/Activate.ps1") {
    Write-Success "Virtual environment found"
    Write-Info "Activating venv..."
    & venv/Scripts/Activate.ps1
} else {
    Write-Warning "Virtual environment not found"
    Write-Info "Creating virtual environment..."
    python -m venv venv
    & venv/Scripts/Activate.ps1
    Write-Info "Installing dependencies..."
    pip install -r requirements.txt
    Write-Success "Dependencies installed"
}

# 3. Data preparation
if (-not $SkipPrep) {
    Write-Info "Checking for training data..."
    
    if (Test-Path $DataDir) {
        $fileCount = (Get-ChildItem -Path $DataDir -Filter "*.txt" -File).Count
        if ($fileCount -gt 0) {
            Write-Success "Found prepared data: $fileCount text files in $DataDir"
            $response = Read-Host "Data already prepared. Re-prepare? (y/N)"
            if ($response -ne "y" -and $response -ne "Y") {
                Write-Info "Skipping data preparation"
            } else {
                Write-Info "Preparing data..."
                python scripts/prepare_data.py --input $DataZip --output $DataDir
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Data preparation failed!"
                    exit 1
                }
            }
        } else {
            Write-Info "Data directory exists but empty. Preparing data..."
            python scripts/prepare_data.py --input $DataZip --output $DataDir
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Data preparation failed!"
                exit 1
            }
        }
    } else {
        Write-Info "No prepared data found. Preparing data..."
        
        if (-not (Test-Path $DataZip)) {
            Write-Error "Data zip not found: $DataZip"
            Write-Info "Please download the dataset and place it at: $DataZip"
            Write-Info "Source: https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish"
            exit 1
        }
        
        python scripts/prepare_data.py --input $DataZip --output $DataDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Data preparation failed!"
            exit 1
        }
    }
} else {
    Write-Info "Skipping data preparation (--SkipPrep flag)"
}

# 4. Launch training
Write-Host @"

================================================================================
STARTING TRAINING
================================================================================
"@ -ForegroundColor Green

Write-Info "Config: $Config"
Write-Info "Data: $DataDir"
Write-Host ""

# Update config to point to correct data directory
# TODO: Could auto-update config file or pass as CLI arg once train.py supports it

python scripts/train.py --config $Config

# Check training result
if ($LASTEXITCODE -eq 0) {
    Write-Host @"

================================================================================
TRAINING COMPLETE
================================================================================
"@ -ForegroundColor Green
    Write-Success "Training finished successfully!"
    Write-Info "Check checkpoints/ for saved models"
    Write-Info "Check training.log for detailed logs"
} else {
    Write-Host @"

================================================================================
TRAINING FAILED
================================================================================
"@ -ForegroundColor Red
    Write-Error "Training exited with error code $LASTEXITCODE"
    Write-Info "Check training.log for details"
    exit 1
}

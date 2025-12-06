#!/usr/bin/env pwsh
# Atlas Complete Training Pipeline
# Handles everything from setup to training start
#
# Usage: .\scripts\start_training.ps1

param(
    [switch]$Help
)

# Color output functions
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Step { Write-Host "`n[STEP] $args" -ForegroundColor Magenta }

# Show help
if ($Help) {
    Write-Host @"
Atlas Complete Training Pipeline

This script handles the entire training pipeline:
1. Checks for training data (archive.zip)
2. Prepares dataset if needed
3. Lets you choose GPU config preset
4. Installs Atlas package if needed
5. Starts training

Usage:
  .\scripts\start_training.ps1

The script will guide you through each step interactively.
"@
    exit 0
}

Write-Host @"
================================================================================
ATLAS TRAINING PIPELINE
================================================================================
This script will guide you through the complete training setup.
"@ -ForegroundColor Cyan

# Step 1: Check Python
Write-Step "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error "Python not found! Please install Python 3.8+"
    Write-Info "Download from: https://www.python.org/downloads/"
    exit 1
}

# Step 2: Check/Create virtual environment
Write-Step "Checking virtual environment..."
if (Test-Path "venv/Scripts/Activate.ps1") {
    Write-Success "Virtual environment found"
    Write-Info "Activating venv..."
    & venv/Scripts/Activate.ps1
} else {
    Write-Warning "Virtual environment not found"
    Write-Info "Creating virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
    & venv/Scripts/Activate.ps1
    Write-Info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies"
        exit 1
    }
    Write-Success "Dependencies installed"
}

# Step 3: Check if Atlas package is installed
Write-Step "Checking Atlas package installation..."
$atlasInstalled = python -c "try: import atlas; print('yes')
except: print('no')" 2>$null

if ($atlasInstalled -eq "no") {
    Write-Warning "Atlas package not installed"
    Write-Info "Installing Atlas in development mode..."
    pip install -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Atlas package"
        exit 1
    }
    Write-Success "Atlas package installed"
} else {
    Write-Success "Atlas package already installed"
}

# Step 4: Check for training data
Write-Step "Checking for training data..."
$dataZip = "data/raw/archive.zip"
$processedDir = "data/processed/wikipedia"

if (-not (Test-Path $dataZip)) {
    Write-Error "Training data not found!"
    Write-Host @"

Please download the Wikipedia SimpleEnglish dataset and place it at:
    $dataZip

Download from: https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish

Steps:
1. Download archive.zip from Kaggle
2. Place it in: data/raw/archive.zip
3. Run this script again

"@ -ForegroundColor Yellow
    exit 1
} else {
    $zipSize = [math]::Round((Get-Item $dataZip).Length / 1MB, 1)
    Write-Success "Found training data: $dataZip ($zipSize MB)"
}

# Step 5: Check if data is already processed
Write-Step "Checking processed data..."
if (Test-Path $processedDir) {
    $fileCount = (Get-ChildItem -Path $processedDir -Filter "*.txt" -File -ErrorAction SilentlyContinue).Count
    if ($fileCount -gt 0) {
        Write-Success "Processed data found: $fileCount text files"
        $response = Read-Host "Data already processed. Re-process? (y/N)"
        if ($response -eq "y" -or $response -eq "Y") {
            Write-Info "Re-processing data..."
            python scripts/prepare_data.py --input $dataZip
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Data preparation failed!"
                exit 1
            }
        } else {
            Write-Info "Using existing processed data"
        }
    } else {
        Write-Info "Processing data for the first time..."
        python scripts/prepare_data.py --input $dataZip
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Data preparation failed!"
            exit 1
        }
    }
} else {
    Write-Info "Processing data for the first time..."
    python scripts/prepare_data.py --input $dataZip
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Data preparation failed!"
        exit 1
    }
}

# Step 6: Choose GPU configuration
Write-Step "Choosing GPU configuration..."
Write-Host @"

Available configurations:

1. SMALL  (~124M params, ~6-8GB VRAM)
   - Fastest training
   - Good for quick experiments
   - Decent quality

2. DEFAULT (~350M params, ~12-14GB VRAM) [RECOMMENDED]
   - Balanced performance
   - Good quality
   - Safe memory margin

3. LARGE  (~500M params, ~14-15GB VRAM)
   - Best quality
   - Slowest training
   - Close to 16GB VRAM limit

"@ -ForegroundColor White

do {
    $choice = Read-Host "Choose configuration (1/2/3)"
} while ($choice -notin @("1", "2", "3"))

$configFile = switch ($choice) {
    "1" { "configs/small.yaml"; $configName = "SMALL" }
    "2" { "configs/default.yaml"; $configName = "DEFAULT" }
    "3" { "configs/large.yaml"; $configName = "LARGE" }
}

Write-Success "Selected: $configName configuration"
Write-Info "Config file: $configFile"

# Verify config file exists
if (-not (Test-Path $configFile)) {
    Write-Error "Configuration file not found: $configFile"
    Write-Info "Please ensure the configs directory is properly set up"
    exit 1
}

# Step 7: Display training information
Write-Step "Training Configuration Summary"
Write-Host @"

Configuration: $configName
Data: $processedDir
Config file: $configFile

The training will:
- Use your RTX 5060 Ti 16GB GPU (if available)
- Save checkpoints to: checkpoints/
- Log to console and: training.log
- Show progress with loss, perplexity, and throughput

You can stop training anytime with Ctrl+C (checkpoint will be saved)

"@ -ForegroundColor White

$response = Read-Host "Ready to start training? (Y/n)"
if ($response -eq "n" -or $response -eq "N") {
    Write-Info "Training cancelled by user"
    exit 0
}

# Step 8: Start training
Write-Step "Starting Training..."
Write-Host @"

================================================================================
TRAINING STARTED
================================================================================

"@ -ForegroundColor Green

python scripts/train.py --config $configFile --train-data $processedDir

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host @"

================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================

"@ -ForegroundColor Green
    Write-Success "Training finished!"
    Write-Info "Checkpoints saved in: checkpoints/"
    Write-Info "Full log available in: training.log"
    Write-Host @"

Next steps:
1. Test your model:
   python scripts/infer.py --checkpoint checkpoints/best_model.pt --prompt "Your prompt here"

2. Export to GGUF:
   python scripts/export_gguf.py --checkpoint checkpoints/best_model.pt --output model.gguf

"@ -ForegroundColor Cyan
} else {
    Write-Host @"

================================================================================
TRAINING FAILED
================================================================================

"@ -ForegroundColor Red
    Write-Error "Training exited with error code $LASTEXITCODE"
    Write-Info "Check training.log for detailed error information"
    exit 1
}

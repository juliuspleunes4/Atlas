#!/usr/bin/env bash
# Atlas Setup and Training Script (Linux/Mac)
# 
# This script automates the entire pipeline:
# 1. Checks prerequisites
# 2. Prepares training data
# 3. Launches training
#
# Usage:
#   ./setup_and_train.sh                    # Use default config
#   ./setup_and_train.sh --config my.yaml   # Use custom config
#   ./setup_and_train.sh --skip-prep        # Skip data preparation

set -e  # Exit on error

# Default parameters
CONFIG="configs/default.yaml"
DATA_ZIP="data/raw/archive.zip"
DATA_DIR="data/processed/wikipedia"
SKIP_PREP=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-zip)
            DATA_ZIP="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip-prep)
            SKIP_PREP=true
            shift
            ;;
        --help|-h)
            cat << EOF
Atlas Setup and Training Script

Usage:
  ./setup_and_train.sh [options]

Options:
  --config <path>     Config file (default: configs/default.yaml)
  --data-zip <path>   Input zip file (default: data/raw/archive.zip)
  --data-dir <path>   Output data directory (default: data/processed/wikipedia)
  --skip-prep         Skip data preparation step
  --help, -h          Show this help message

Examples:
  # Full pipeline (prepare data + train)
  ./setup_and_train.sh

  # Use custom config
  ./setup_and_train.sh --config configs/large.yaml

  # Skip data preparation (data already prepared)
  ./setup_and_train.sh --skip-prep

  # Custom data source
  ./setup_and_train.sh --data-zip data/raw/my_data.zip --data-dir data/processed/my_data
EOF
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}================================================================================"
echo "ATLAS SETUP AND TRAINING"
echo -e "================================================================================${NC}"

# 1. Check Python
info "Checking Python..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found! Please install Python 3.8+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
success "Python: $PYTHON_VERSION"

# 2. Check virtual environment
info "Checking virtual environment..."
if [ -d "venv" ]; then
    success "Virtual environment found"
    info "Activating venv..."
    source venv/bin/activate
else
    warning "Virtual environment not found"
    info "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    info "Installing dependencies..."
    pip install -r requirements.txt
    success "Dependencies installed"
fi

# 3. Data preparation
if [ "$SKIP_PREP" = false ]; then
    info "Checking for training data..."
    
    if [ -d "$DATA_DIR" ]; then
        FILE_COUNT=$(find "$DATA_DIR" -name "*.txt" -type f | wc -l)
        if [ "$FILE_COUNT" -gt 0 ]; then
            success "Found prepared data: $FILE_COUNT text files in $DATA_DIR"
            read -p "Data already prepared. Re-prepare? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                info "Skipping data preparation"
            else
                info "Preparing data..."
                python scripts/prepare_data.py --input "$DATA_ZIP" --output "$DATA_DIR"
            fi
        else
            info "Data directory exists but empty. Preparing data..."
            python scripts/prepare_data.py --input "$DATA_ZIP" --output "$DATA_DIR"
        fi
    else
        info "No prepared data found. Preparing data..."
        
        if [ ! -f "$DATA_ZIP" ]; then
            error "Data zip not found: $DATA_ZIP"
            info "Please download the dataset and place it at: $DATA_ZIP"
            info "Source: https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish"
            exit 1
        fi
        
        python scripts/prepare_data.py --input "$DATA_ZIP" --output "$DATA_DIR"
    fi
else
    info "Skipping data preparation (--skip-prep flag)"
fi

# 4. Launch training
echo -e "\n${GREEN}================================================================================"
echo "STARTING TRAINING"
echo -e "================================================================================${NC}\n"

info "Config: $CONFIG"
info "Data: $DATA_DIR"
echo

# Update config to point to correct data directory
# TODO: Could auto-update config file or pass as CLI arg once train.py supports it

python scripts/train.py --config "$CONFIG"

# Check training result
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================================================"
    echo "TRAINING COMPLETE"
    echo -e "================================================================================${NC}\n"
    success "Training finished successfully!"
    info "Check checkpoints/ for saved models"
    info "Check training.log for detailed logs"
else
    echo -e "\n${RED}================================================================================"
    echo "TRAINING FAILED"
    echo -e "================================================================================${NC}\n"
    error "Training exited with error code $?"
    info "Check training.log for details"
    exit 1
fi

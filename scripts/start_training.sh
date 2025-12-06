#!/usr/bin/env bash
# Atlas Complete Training Pipeline
# Handles everything from setup to training start
#
# Usage: ./scripts/start_training.sh

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
step() { echo -e "\n${MAGENTA}[STEP] $1${NC}"; }

# Show help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    cat << EOF
Atlas Complete Training Pipeline

This script handles the entire training pipeline:
1. Checks for training data (archive.zip)
2. Prepares dataset if needed
3. Lets you choose GPU config preset
4. Installs Atlas package if needed
5. Starts training

Usage:
  ./scripts/start_training.sh

The script will guide you through each step interactively.
EOF
    exit 0
fi

echo -e "${CYAN}================================================================================"
echo "ATLAS TRAINING PIPELINE"
echo "================================================================================"
echo "This script will guide you through the complete training setup.${NC}"

# Step 1: Check Python
step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found! Please install Python 3.8+"
    info "Visit: https://www.python.org/downloads/"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
success "Python found: $PYTHON_VERSION"

# Step 2: Check/Create virtual environment
step "Checking virtual environment..."
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
    pip install --upgrade pip
    pip install -r requirements.txt
    success "Dependencies installed"
fi

# Step 3: Check if Atlas package is installed
step "Checking Atlas package installation..."
if ! python -c "import atlas" 2>/dev/null; then
    warning "Atlas package not installed"
    info "Installing Atlas in development mode..."
    pip install -e .
    success "Atlas package installed"
else
    success "Atlas package already installed"
fi

# Step 4: Check for training data
step "Checking for training data..."
DATA_ZIP="data/raw/archive.zip"
PROCESSED_DIR="data/processed/wikipedia"

if [ ! -f "$DATA_ZIP" ]; then
    error "Training data not found!"
    echo -e "${YELLOW}
Please download the Wikipedia SimpleEnglish dataset and place it at:
    $DATA_ZIP

Download from: https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish

Steps:
1. Download archive.zip from Kaggle
2. Place it in: data/raw/archive.zip
3. Run this script again
${NC}"
    exit 1
else
    ZIP_SIZE=$(du -h "$DATA_ZIP" | cut -f1)
    success "Found training data: $DATA_ZIP ($ZIP_SIZE)"
fi

# Step 5: Check if data is already processed
step "Checking processed data..."
if [ -d "$PROCESSED_DIR" ]; then
    FILE_COUNT=$(find "$PROCESSED_DIR" -name "*.txt" -type f 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        success "Processed data found: $FILE_COUNT text files"
        read -p "Data already processed. Re-process? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Re-processing data..."
            python scripts/prepare_data.py --input "$DATA_ZIP"
        else
            info "Using existing processed data"
        fi
    else
        info "Processing data for the first time..."
        python scripts/prepare_data.py --input "$DATA_ZIP"
    fi
else
    info "Processing data for the first time..."
    python scripts/prepare_data.py --input "$DATA_ZIP"
fi

# Step 6: Choose GPU configuration
step "Choosing GPU configuration..."
cat << EOF

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

EOF

while true; do
    read -p "Choose configuration (1/2/3): " choice
    case $choice in
        1) CONFIG_FILE="configs/small.yaml"; CONFIG_NAME="SMALL"; break;;
        2) CONFIG_FILE="configs/default.yaml"; CONFIG_NAME="DEFAULT"; break;;
        3) CONFIG_FILE="configs/large.yaml"; CONFIG_NAME="LARGE"; break;;
        *) warning "Invalid choice. Please enter 1, 2, or 3.";;
    esac
done

success "Selected: $CONFIG_NAME configuration"
info "Config file: $CONFIG_FILE"

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    error "Configuration file not found: $CONFIG_FILE"
    info "Please ensure the configs directory is properly set up"
    exit 1
fi

# Step 7: Display training information
step "Training Configuration Summary"
cat << EOF

Configuration: $CONFIG_NAME
Data: $PROCESSED_DIR
Config file: $CONFIG_FILE

The training will:
- Use your GPU (if available)
- Save checkpoints to: checkpoints/
- Log to console and: training.log
- Show progress with loss, perplexity, and throughput

You can stop training anytime with Ctrl+C (checkpoint will be saved)

EOF

read -p "Ready to start training? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    info "Training cancelled by user"
    exit 0
fi

# Step 8: Start training
step "Starting Training..."
echo -e "${GREEN}
================================================================================
TRAINING STARTED
================================================================================
${NC}"

python scripts/train.py --config "$CONFIG_FILE" --train-data "$PROCESSED_DIR"

# Check result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}
================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================
${NC}"
    success "Training finished!"
    info "Checkpoints saved in: checkpoints/"
    info "Full log available in: training.log"
    echo -e "${CYAN}
Next steps:
1. Test your model:
   python scripts/infer.py --checkpoint checkpoints/best_model.pt --prompt \"Your prompt here\"

2. Export to GGUF:
   python scripts/export_gguf.py --checkpoint checkpoints/best_model.pt --output model.gguf
${NC}"
else
    echo -e "${RED}
================================================================================
TRAINING FAILED
================================================================================
${NC}"
    error "Training exited with error code $?"
    info "Check training.log for detailed error information"
    exit 1
fi

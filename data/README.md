# Atlas Data Directory

This directory contains training data for Atlas. **All contents are gitignored** to avoid committing large datasets.

## Structure

```
data/
├── raw/           # Original, unprocessed datasets
│   └── archive.zip    # Wikipedia SimpleEnglish zip from Kaggle
├── processed/     # Tokenized and prepared training data
│   └── wikipedia/ # Extracted and processed Wikipedia articles
└── README.md      # This file
```

## Current Datasets

### Wikipedia SimpleEnglish

**Source**: [Kaggle - Plain text Wikipedia (SimpleEnglish)](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish)

**Stats**:
- 249,396 articles
- 31M tokens
- 196,000 words
- ~400 MB uncompressed
- 171 MB zip file

**Location**: Place `archive.zip` in `data/raw/`

**Format**:
- Each article's title appears before the content
- Articles are plain text (stripped of Wiki formatting)
- Articles concatenated into txt files of ≤ 1MB each
- Clean and uniform data

## Usage

### Quick Start (Automated)

**One command to rule them all:**

```powershell
# Windows
.\scripts\setup_and_train.ps1

# Linux/Mac
./scripts/setup_and_train.sh
```

This handles everything: data prep, training, logging.

### Manual Steps

1. **Place raw data**: 
   ```
   data/raw/archive.zip
   ```

2. **Prepare data**:
   ```bash
   python scripts/prepare_data.py --input data/raw/archive.zip
   ```
   
   Output: `data/processed/wikipedia/` with organized text files

3. **Verify prepared data**:
   ```bash
   python scripts/prepare_data.py --list
   ```

4. **Train**:
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

### Advanced Options

```bash
# Custom output directory
python scripts/prepare_data.py \
  --input data/raw/archive.zip \
  --output data/processed/my_wiki

# Skip data prep (already done)
.\scripts\setup_and_train.ps1 -SkipPrep

# Custom config
.\scripts\setup_and_train.ps1 -Config configs/large.yaml
```

## Notes

- All data files are gitignored (`.txt`, `.jsonl`, `.csv`, `.tsv`, `.zip`)
- Keep raw archives in `data/raw/` for reproducibility
- Processed data goes in `data/processed/`
- Document any new datasets in this README

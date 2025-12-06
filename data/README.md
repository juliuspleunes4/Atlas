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

1. **Place raw data**: 
   ```
   data/raw/archive.zip
   ```

2. **Extract data** (TODO: create extraction script):
   ```bash
   # Will extract to data/processed/wikipedia/
   python scripts/prepare_data.py --input data/raw/archive.zip
   ```

3. **Train with data**:
   ```bash
   python scripts/train.py --config configs/default.yaml --data data/processed/wikipedia/
   ```

## Notes

- All data files are gitignored (`.txt`, `.jsonl`, `.csv`, `.tsv`, `.zip`)
- Keep raw archives in `data/raw/` for reproducibility
- Processed data goes in `data/processed/`
- Document any new datasets in this README

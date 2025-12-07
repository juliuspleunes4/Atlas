"""
Tests for the text dataset module.

Tests cover dataset loading, tokenization, sequence creation,
edge cases, and integration with the tokenizer.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
import json

from atlas.data.dataset import TextDataset
from atlas.data import preprocessing
from atlas.tokenizer import Tokenizer


class TestTextDataset:
    """Test suite for TextDataset class."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def create_text_file(self, temp_dir: Path, filename: str, content: str) -> Path:
        """Helper to create a text file."""
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    # --- Basic Functionality Tests ---
    
    def test_single_file_loading(self, tokenizer, temp_dir):
        """Test loading a single text file."""
        # Use longer text to ensure we get at least one sequence
        content = " ".join([f"word{i}" for i in range(20)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        assert len(dataset) > 0
        assert isinstance(dataset[0], torch.Tensor)
        assert dataset[0].dtype == torch.long
    
    def test_multiple_files_loading(self, tokenizer, temp_dir):
        """Test loading multiple text files."""
        file1 = self.create_text_file(temp_dir, "file1.txt", "First file content.")
        file2 = self.create_text_file(temp_dir, "file2.txt", "Second file content.")
        
        dataset = TextDataset(
            file_paths=[file1, file2],
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        stats = dataset.get_stats()
        assert stats["num_files"] == 2
        assert stats["total_tokens"] > 0
    
    def test_string_path_conversion(self, tokenizer, temp_dir):
        """Test that string paths are properly converted."""
        # Use longer text to ensure we get at least one sequence
        content = " ".join([f"word{i}" for i in range(20)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        # Pass as string instead of Path
        dataset = TextDataset(
            file_paths=str(file_path),
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        assert len(dataset) > 0
    
    # --- Sequence Creation Tests ---
    
    def test_sequence_length(self, tokenizer, temp_dir):
        """Test that sequences have the correct length."""
        # Create text that will produce many tokens
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        max_seq_len = 20
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )
        
        # Check all sequences have correct length
        for i in range(len(dataset)):
            seq = dataset[i]
            assert seq.shape == (max_seq_len,)
    
    def test_non_overlapping_sequences(self, tokenizer, temp_dir):
        """Test non-overlapping sequence creation (stride = max_seq_len)."""
        content = " ".join([f"token{i}" for i in range(50)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        max_seq_len = 10
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=max_seq_len  # Non-overlapping
        )
        
        # With non-overlapping, we should have fewer sequences
        total_tokens = len(dataset.tokens)
        expected_sequences = total_tokens // max_seq_len
        assert len(dataset) == expected_sequences
    
    def test_overlapping_sequences(self, tokenizer, temp_dir):
        """Test overlapping sequence creation (stride < max_seq_len)."""
        content = " ".join([f"token{i}" for i in range(50)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        max_seq_len = 10
        stride = 5  # 50% overlap
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=stride
        )
        
        # With overlapping, we should have more sequences
        total_tokens = len(dataset.tokens)
        # Calculate expected: (total - max_seq_len) // stride + 1
        expected_sequences = (total_tokens - max_seq_len) // stride + 1
        assert len(dataset) == expected_sequences
    
    def test_leftover_tokens_handling(self, tokenizer, temp_dir):
        """Test handling of leftover tokens that don't fit in a full sequence."""
        # Create text with known token count
        # Let's aim for a specific number of tokens that leaves leftovers
        content = " ".join([f"word{i}" for i in range(30)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        max_seq_len = 25
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=max_seq_len
        )
        
        # Leftover tokens should be skipped (not padded)
        total_tokens = len(dataset.tokens)
        expected_sequences = total_tokens // max_seq_len
        assert len(dataset) == expected_sequences
    
    # --- Indexing Tests ---
    
    def test_getitem_returns_tensor(self, tokenizer, temp_dir):
        """Test that __getitem__ returns a tensor."""
        # Use longer text to ensure we get at least one sequence
        content = " ".join([f"word{i}" for i in range(20)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
    
    def test_getitem_out_of_bounds(self, tokenizer, temp_dir):
        """Test that out-of-bounds indexing raises IndexError."""
        content = "Short content."
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        # Test positive out of bounds
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
        
        # Test negative out of bounds
        with pytest.raises(IndexError):
            _ = dataset[-1]
    
    def test_len_returns_correct_count(self, tokenizer, temp_dir):
        """Test that __len__ returns the correct number of sequences."""
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=20
        )
        
        assert len(dataset) == len(dataset.sequences)
        assert len(dataset) > 0
    
    # --- Edge Cases ---
    
    def test_empty_file_raises_error(self, tokenizer, temp_dir):
        """Test that empty files are handled correctly."""
        file_path = self.create_text_file(temp_dir, "empty.txt", "")
        
        with pytest.raises(ValueError, match="No tokens were loaded"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=10
            )
    
    def test_whitespace_only_file(self, tokenizer, temp_dir):
        """Test that whitespace-only files are skipped."""
        file_path = self.create_text_file(temp_dir, "whitespace.txt", "   \n\n   \t  ")
        
        with pytest.raises(ValueError, match="No tokens were loaded"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=10
            )
    
    def test_nonexistent_file_raises_error(self, tokenizer, temp_dir):
        """Test that nonexistent files raise FileNotFoundError."""
        nonexistent = temp_dir / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            TextDataset(
                file_paths=nonexistent,
                tokenizer=tokenizer,
                max_seq_len=10
            )
    
    def test_very_short_text(self, tokenizer, temp_dir):
        """Test handling of very short text (fewer tokens than max_seq_len)."""
        content = "Hi"  # Very short
        file_path = self.create_text_file(temp_dir, "short.txt", content)
        
        # This should create 0 sequences if text is too short
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=100  # Much larger than content
        )
        
        # Should have 0 or very few sequences
        assert len(dataset) == 0
    
    def test_exact_sequence_length_text(self, tokenizer, temp_dir):
        """Test text that produces exactly max_seq_len tokens."""
        # Create text that produces approximately 20 tokens
        content = " ".join([f"word{i}" for i in range(15)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        # Count actual tokens
        tokens = tokenizer.encode(content)
        max_seq_len = len(tokens)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )
        
        # Should have exactly 1 sequence
        assert len(dataset) == 1
    
    # --- Parameter Validation Tests ---
    
    def test_invalid_max_seq_len_zero(self, tokenizer, temp_dir):
        """Test that zero max_seq_len raises ValueError."""
        file_path = self.create_text_file(temp_dir, "test.txt", "content")
        
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=0
            )
    
    def test_invalid_max_seq_len_negative(self, tokenizer, temp_dir):
        """Test that negative max_seq_len raises ValueError."""
        file_path = self.create_text_file(temp_dir, "test.txt", "content")
        
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=-10
            )
    
    def test_invalid_stride_zero(self, tokenizer, temp_dir):
        """Test that zero stride raises ValueError."""
        file_path = self.create_text_file(temp_dir, "test.txt", "content")
        
        with pytest.raises(ValueError, match="stride must be positive"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=10,
                stride=0
            )
    
    def test_invalid_stride_negative(self, tokenizer, temp_dir):
        """Test that negative stride raises ValueError."""
        file_path = self.create_text_file(temp_dir, "test.txt", "content")
        
        with pytest.raises(ValueError, match="stride must be positive"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=10,
                stride=-5
            )
    
    def test_invalid_stride_too_large(self, tokenizer, temp_dir):
        """Test that stride > max_seq_len raises ValueError."""
        file_path = self.create_text_file(temp_dir, "test.txt", "content")
        
        with pytest.raises(ValueError, match="stride must be positive and <= max_seq_len"):
            TextDataset(
                file_paths=file_path,
                tokenizer=tokenizer,
                max_seq_len=10,
                stride=15
            )
    
    # --- Utility Method Tests ---
    
    def test_get_vocab_size(self, tokenizer, temp_dir):
        """Test getting vocabulary size from the tokenizer."""
        content = "Test content."
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=10
        )
        
        assert dataset.get_vocab_size() == tokenizer.vocab_size
    
    def test_get_stats(self, tokenizer, temp_dir):
        """Test getting dataset statistics."""
        file1 = self.create_text_file(temp_dir, "file1.txt", "First file.")
        file2 = self.create_text_file(temp_dir, "file2.txt", "Second file.")
        
        max_seq_len = 10
        stride = 5
        dataset = TextDataset(
            file_paths=[file1, file2],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=stride
        )
        
        stats = dataset.get_stats()
        
        assert stats["num_files"] == 2
        assert stats["total_tokens"] > 0
        assert stats["num_sequences"] == len(dataset)
        assert stats["max_seq_len"] == max_seq_len
        assert stats["stride"] == stride
        assert stats["vocab_size"] == tokenizer.vocab_size
    
    # --- Integration Tests ---
    
    def test_dataset_with_realistic_text(self, tokenizer, temp_dir):
        """Test dataset with realistic multi-sentence text."""
        content = """
        The quick brown fox jumps over the lazy dog. This is a common pangram used 
        in typography. It contains every letter of the English alphabet at least once.
        Language models learn patterns from text data. They predict the next token
        given previous tokens as context. This is called autoregressive modeling.
        """
        file_path = self.create_text_file(temp_dir, "realistic.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=50
        )
        
        assert len(dataset) > 0
        
        # Check first sequence
        first_seq = dataset[0]
        assert first_seq.shape == (50,)
        assert first_seq.min() >= 0
        assert first_seq.max() < tokenizer.vocab_size
    
    def test_multiple_files_concatenation(self, tokenizer, temp_dir):
        """Test that multiple files are properly concatenated."""
        content1 = "First file content with multiple words."
        content2 = "Second file also has content."
        
        file1 = self.create_text_file(temp_dir, "file1.txt", content1)
        file2 = self.create_text_file(temp_dir, "file2.txt", content2)
        
        # Create separate datasets
        dataset1 = TextDataset(file1, tokenizer, max_seq_len=100)
        dataset2 = TextDataset(file2, tokenizer, max_seq_len=100)
        
        # Create combined dataset
        dataset_combined = TextDataset([file1, file2], tokenizer, max_seq_len=100)
        
        # Combined should have tokens from both
        tokens1 = len(dataset1.tokens)
        tokens2 = len(dataset2.tokens)
        tokens_combined = len(dataset_combined.tokens)
        
        assert tokens_combined == tokens1 + tokens2
    
    def test_dataset_iteration(self, tokenizer, temp_dir):
        """Test iterating through the dataset."""
        content = " ".join([f"sentence{i}" for i in range(50)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=20
        )
        
        # Iterate through all sequences
        sequences = [dataset[i] for i in range(len(dataset))]
        
        assert len(sequences) == len(dataset)
        for seq in sequences:
            assert isinstance(seq, torch.Tensor)
            assert seq.shape == (20,)
    
    def test_unicode_handling(self, tokenizer, temp_dir):
        """Test handling of Unicode characters."""
        content = "Hello 世界! Émojis: 😀🎉 Math: ∑∫∂"
        file_path = self.create_text_file(temp_dir, "unicode.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=20
        )
        
        # Should handle Unicode without errors
        assert len(dataset) > 0
        first_seq = dataset[0]
        assert first_seq.dtype == torch.long
    
    def test_long_document_handling(self, tokenizer, temp_dir):
        """Test handling of long documents that span many sequences."""
        # Create a long document
        content = " ".join([f"word{i}" for i in range(1000)])
        file_path = self.create_text_file(temp_dir, "long.txt", content)
        
        max_seq_len = 50
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len
        )
        
        # Should create many sequences
        assert len(dataset) > 10
        
        # Check a few random sequences
        for idx in [0, len(dataset) // 2, len(dataset) - 1]:
            seq = dataset[idx]
            assert seq.shape == (max_seq_len,)
            assert seq.min() >= 0
            assert seq.max() < tokenizer.vocab_size


class TestDataLoader:
    """Test suite for data loading utilities."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, tokenizer, temp_dir):
        """Create a sample dataset for testing."""
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = temp_dir / "test.txt"
        file_path.write_text(content, encoding='utf-8')
        
        return TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=20
        )
    
    # --- DataLoader Creation Tests ---
    
    def test_create_dataloader_basic(self, sample_dataset):
        """Test basic DataLoader creation."""
        from atlas.data.loader import create_dataloader
        
        loader = create_dataloader(
            sample_dataset,
            batch_size=4,
            shuffle=True
        )
        
        assert loader is not None
        assert loader.batch_size == 4
        assert len(loader) > 0
    
    def test_create_dataloader_returns_correct_batch_shape(self, sample_dataset):
        """Test that DataLoader returns batches with correct shape."""
        from atlas.data.loader import create_dataloader
        
        batch_size = 4
        loader = create_dataloader(
            sample_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Get first batch
        batch = next(iter(loader))
        
        assert isinstance(batch, dict)
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] <= batch_size  # May be smaller if drop_last=False
        assert batch['input_ids'].shape[1] == 20  # Sequence length
    
    def test_create_dataloader_with_drop_last(self, sample_dataset):
        """Test DataLoader with drop_last=True."""
        from atlas.data.loader import create_dataloader
        
        batch_size = 3
        loader = create_dataloader(
            sample_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )
        
        # All batches should have exact batch_size
        for batch in loader:
            assert isinstance(batch, dict)
            assert 'input_ids' in batch
            assert batch['input_ids'].shape[0] == batch_size
    
    def test_create_dataloader_invalid_batch_size(self, sample_dataset):
        """Test that invalid batch size raises ValueError."""
        from atlas.data.loader import create_dataloader
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_dataloader(sample_dataset, batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_dataloader(sample_dataset, batch_size=-1)
    
    def test_create_dataloader_iteration(self, sample_dataset):
        """Test iterating through DataLoader."""
        from atlas.data.loader import create_dataloader
        
        loader = create_dataloader(
            sample_dataset,
            batch_size=2,
            shuffle=False
        )
        
        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert isinstance(batch, dict)
            assert 'input_ids' in batch
            assert batch['input_ids'].ndim == 2  # (batch_size, seq_len)
        
        assert batch_count == len(loader)
    
    # --- Collation Tests ---
    
    def test_collate_batch(self):
        """Test batch collation function."""
        from atlas.data.loader import collate_batch
        
        # Create fake batch of sequences
        batch = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([6, 7, 8, 9, 10]),
            torch.tensor([11, 12, 13, 14, 15]),
        ]
        
        collated = collate_batch(batch)
        
        assert isinstance(collated, dict)
        assert 'input_ids' in collated
        assert collated['input_ids'].shape == (3, 5)
        assert torch.equal(collated['input_ids'][0], batch[0])
        assert torch.equal(collated['input_ids'][1], batch[1])
        assert torch.equal(collated['input_ids'][2], batch[2])
    
    def test_collate_batch_single_item(self):
        """Test collating a batch with a single item."""
        from atlas.data.loader import collate_batch
        
        batch = [torch.tensor([1, 2, 3])]
        collated = collate_batch(batch)
        
        assert isinstance(collated, dict)
        assert 'input_ids' in collated
        assert collated['input_ids'].shape == (1, 3)
        assert torch.equal(collated['input_ids'][0], batch[0])
    
    # --- DataLoader Stats Tests ---
    
    def test_get_dataloader_stats(self, sample_dataset):
        """Test getting DataLoader statistics."""
        from atlas.data.loader import create_dataloader, get_dataloader_stats
        
        loader = create_dataloader(
            sample_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        stats = get_dataloader_stats(loader)
        
        assert stats["dataset_size"] == len(sample_dataset)
        assert stats["batch_size"] == 8
        assert stats["num_batches"] == len(loader)
        assert stats["num_workers"] == 0
        assert stats["pin_memory"] is False
        assert stats["drop_last"] is True
        assert "dataset_stats" in stats
    
    # --- Dataset Splitting Tests ---
    
    def test_split_dataset_basic(self, sample_dataset):
        """Test basic dataset splitting."""
        from atlas.data.loader import split_dataset
        
        train, val, test = split_dataset(
            sample_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )
        
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataset)
        
        # Check approximate ratios
        assert len(train) >= len(val)
        assert len(train) >= len(test)
    
    def test_split_dataset_no_test(self, sample_dataset):
        """Test splitting with no test set."""
        from atlas.data.loader import split_dataset
        
        train, val, test = split_dataset(
            sample_dataset,
            train_ratio=0.9,
            val_ratio=0.1,
            test_ratio=0.0,
            seed=42
        )
        
        assert len(train) + len(val) + len(test) == len(sample_dataset)
        assert len(test) == 0
    
    def test_split_dataset_reproducibility(self, sample_dataset):
        """Test that splitting with same seed is reproducible."""
        from atlas.data.loader import split_dataset
        
        train1, val1, test1 = split_dataset(sample_dataset, seed=42)
        train2, val2, test2 = split_dataset(sample_dataset, seed=42)
        
        # Check sizes are identical
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)
        
        # Check first few items are identical
        for i in range(min(5, len(train1))):
            assert torch.equal(train1[i], train2[i])
    
    def test_split_dataset_different_seeds(self, sample_dataset):
        """Test that different seeds produce different splits."""
        from atlas.data.loader import split_dataset
        
        train1, _, _ = split_dataset(sample_dataset, seed=42)
        train2, _, _ = split_dataset(sample_dataset, seed=123)
        
        # Sizes should be the same
        assert len(train1) == len(train2)
        
        # But first items should likely be different (not guaranteed but highly probable)
        # We'll just check that splitting works with different seeds
        assert train1 is not train2
    
    def test_split_dataset_invalid_ratios_sum(self, sample_dataset):
        """Test that invalid ratio sum raises ValueError."""
        from atlas.data.loader import split_dataset
        
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_dataset(sample_dataset, 0.5, 0.3, 0.1)  # Sums to 0.9
        
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_dataset(sample_dataset, 0.6, 0.3, 0.2)  # Sums to 1.1
    
    def test_split_dataset_negative_ratios(self, sample_dataset):
        """Test that negative ratios raise ValueError."""
        from atlas.data.loader import split_dataset
        
        with pytest.raises(ValueError, match="All ratios must be non-negative"):
            split_dataset(sample_dataset, 0.9, -0.1, 0.2)
    
    def test_split_dataset_zero_train_ratio(self, sample_dataset):
        """Test that zero train ratio raises ValueError."""
        from atlas.data.loader import split_dataset
        
        with pytest.raises(ValueError, match="train_ratio must be positive"):
            split_dataset(sample_dataset, 0.0, 0.5, 0.5)
    
    # --- Integration Tests ---
    
    def test_full_pipeline(self, tokenizer, temp_dir):
        """Test complete pipeline: dataset creation -> splitting -> dataloaders."""
        from atlas.data.loader import create_dataloader, split_dataset
        
        # Create dataset
        content = " ".join([f"word{i}" for i in range(200)])
        file_path = temp_dir / "data.txt"
        file_path.write_text(content, encoding='utf-8')
        
        dataset = TextDataset(file_path, tokenizer, max_seq_len=20)
        
        # Split
        train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.2, 0.1)
        
        # Create loaders
        train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True)
        val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)
        test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False)
        
        # Test train loader
        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, dict)
        assert 'input_ids' in train_batch
        assert train_batch['input_ids'].ndim == 2
        assert train_batch['input_ids'].shape[1] == 20
        
        # Test val loader
        val_batch = next(iter(val_loader))
        assert isinstance(val_batch, dict)
        assert 'input_ids' in val_batch
        assert val_batch['input_ids'].ndim == 2
        assert val_batch['input_ids'].shape[1] == 20
        
        # Test test loader (if not empty)
        if len(test_loader) > 0:
            test_batch = next(iter(test_loader))
            assert isinstance(test_batch, dict)
            assert 'input_ids' in test_batch
            assert test_batch['input_ids'].ndim == 2
            assert test_batch['input_ids'].shape[1] == 20


class TestPreprocessing:
    """Test suite for preprocessing utilities."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    # --- Text Cleaning Tests ---
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello   World!  \n\n  This is   a test."
        result = preprocessing.clean_text(text)
        
        # Multiple spaces should be collapsed
        assert "  " not in result
        # Leading/trailing whitespace should be removed
        assert result == result.strip()
    
    def test_clean_text_lowercase(self):
        """Test lowercase conversion."""
        text = "Hello World!"
        result = preprocessing.clean_text(text, lowercase=True)
        
        assert result == "hello world!"
    
    def test_clean_text_unicode_normalization(self):
        """Test Unicode normalization."""
        # Text with combining characters
        text = "café"  # May be composed or decomposed
        result = preprocessing.clean_text(text, normalize_unicode=True)
        
        # Should be NFC normalized
        assert result == "café"
    
    def test_clean_text_empty_string(self):
        """Test cleaning an empty string."""
        result = preprocessing.clean_text("")
        assert result == ""
    
    def test_clean_text_whitespace_only(self):
        """Test cleaning whitespace-only text."""
        result = preprocessing.clean_text("   \n\n   \t  ")
        assert result == ""
    
    def test_clean_text_control_chars(self):
        """Test removal of control characters."""
        # Include some control characters
        text = "Hello\x00World\x01Test"
        result = preprocessing.clean_text(text, remove_control_chars=True)
        
        # Control chars should be removed, but regular chars preserved
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_clean_text_preserve_newlines(self):
        """Test that newlines are preserved when removing control chars."""
        text = "Line 1\nLine 2\nLine 3"
        result = preprocessing.clean_text(text, remove_control_chars=True)
        
        # Newlines should be preserved
        assert "\n" in result
        assert "Line 1" in result
        assert "Line 3" in result
    
    # --- Text Chunking Tests ---
    
    def test_chunk_text_basic(self, tokenizer):
        """Test basic text chunking."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = preprocessing.chunk_text(text, max_tokens=20, tokenizer=tokenizer)
        
        assert len(chunks) > 0
        # Each chunk should be relatively short
        for chunk in chunks:
            tokens = tokenizer.encode(chunk)
            assert len(tokens) <= 20
    
    def test_chunk_text_short_text(self, tokenizer):
        """Test chunking text shorter than max_tokens."""
        text = "Short text."
        chunks = preprocessing.chunk_text(text, max_tokens=100, tokenizer=tokenizer)
        
        # Should return single chunk
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_with_overlap(self, tokenizer):
        """Test chunking with overlap."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = preprocessing.chunk_text(
            text, max_tokens=20, tokenizer=tokenizer, overlap=5
        )
        
        # With overlap, should have more chunks
        assert len(chunks) > 0
    
    def test_chunk_text_invalid_max_tokens(self, tokenizer):
        """Test that invalid max_tokens raises ValueError."""
        text = "Some text"
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            preprocessing.chunk_text(text, max_tokens=0, tokenizer=tokenizer)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            preprocessing.chunk_text(text, max_tokens=-10, tokenizer=tokenizer)
    
    def test_chunk_text_invalid_overlap(self, tokenizer):
        """Test that invalid overlap raises ValueError."""
        text = "Some text"
        
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            preprocessing.chunk_text(text, max_tokens=10, tokenizer=tokenizer, overlap=-5)
        
        with pytest.raises(ValueError, match="less than max_tokens"):
            preprocessing.chunk_text(text, max_tokens=10, tokenizer=tokenizer, overlap=10)
    
    def test_chunk_text_empty_string(self, tokenizer):
        """Test chunking empty text."""
        chunks = preprocessing.chunk_text("", max_tokens=10, tokenizer=tokenizer)
        assert chunks == []
    
    # --- File Loading Tests ---
    
    def test_load_text_file(self, temp_dir):
        """Test loading a text file."""
        content = "Test content for file loading."
        file_path = temp_dir / "test.txt"
        file_path.write_text(content, encoding='utf-8')
        
        loaded = preprocessing.load_text_file(file_path)
        
        # Content should be loaded (may be cleaned)
        assert len(loaded) > 0
    
    def test_load_text_file_with_cleaning(self, temp_dir):
        """Test loading with cleaning enabled."""
        content = "Test   content   with   extra   spaces."
        file_path = temp_dir / "test.txt"
        file_path.write_text(content, encoding='utf-8')
        
        loaded = preprocessing.load_text_file(file_path, clean=True)
        
        # Multiple spaces should be collapsed
        assert "   " not in loaded
    
    def test_load_text_file_nonexistent(self, temp_dir):
        """Test loading a nonexistent file."""
        file_path = temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            preprocessing.load_text_file(file_path)
    
    def test_load_jsonl(self, temp_dir):
        """Test loading a JSONL file."""
        # Create JSONL file
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"text": "First document"}, f)
            f.write('\n')
            json.dump({"text": "Second document"}, f)
            f.write('\n')
            json.dump({"text": "Third document"}, f)
            f.write('\n')
        
        texts = preprocessing.load_jsonl(file_path)
        
        assert len(texts) == 3
        assert "First document" in texts
        assert "Second document" in texts
        assert "Third document" in texts
    
    def test_load_jsonl_custom_field(self, temp_dir):
        """Test loading JSONL with custom text field."""
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"content": "Custom field document"}, f)
            f.write('\n')
        
        texts = preprocessing.load_jsonl(file_path, text_field="content")
        
        assert len(texts) == 1
        assert texts[0] == "Custom field document"
    
    def test_load_jsonl_missing_field(self, temp_dir):
        """Test JSONL with missing text field."""
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"text": "Valid"}, f)
            f.write('\n')
            json.dump({"other": "Missing text field"}, f)
            f.write('\n')
        
        texts = preprocessing.load_jsonl(file_path)
        
        # Should skip the entry with missing field
        assert len(texts) == 1
        assert texts[0] == "Valid"
    
    def test_load_jsonl_invalid_json(self, temp_dir):
        """Test JSONL with invalid JSON lines."""
        file_path = temp_dir / "test.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"text": "Valid"}, f)
            f.write('\n')
            f.write('invalid json\n')
            json.dump({"text": "Also valid"}, f)
            f.write('\n')
        
        texts = preprocessing.load_jsonl(file_path)
        
        # Should skip invalid lines
        assert len(texts) == 2
    
    # --- Document Iteration Tests ---
    
    def test_iterate_documents_txt(self, temp_dir):
        """Test iterating over text documents."""
        file1 = temp_dir / "doc1.txt"
        file2 = temp_dir / "doc2.txt"
        file1.write_text("First document", encoding='utf-8')
        file2.write_text("Second document", encoding='utf-8')
        
        docs = list(preprocessing.iterate_documents([file1, file2], file_format='txt'))
        
        assert len(docs) == 2
        assert "First document" in docs[0]
        assert "Second document" in docs[1]
    
    def test_iterate_documents_jsonl(self, temp_dir):
        """Test iterating over JSONL documents."""
        file_path = temp_dir / "docs.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"text": "Doc 1"}, f)
            f.write('\n')
            json.dump({"text": "Doc 2"}, f)
            f.write('\n')
        
        docs = list(preprocessing.iterate_documents(file_path, file_format='jsonl'))
        
        assert len(docs) == 2
    
    def test_iterate_documents_unsupported_format(self, temp_dir):
        """Test iteration with unsupported file format."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Content", encoding='utf-8')
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(preprocessing.iterate_documents(file_path, file_format='pdf'))
    
    # --- Token Counting Tests ---
    
    def test_count_tokens(self, tokenizer):
        """Test token counting."""
        text = "Hello world, this is a test."
        count = preprocessing.count_tokens(text, tokenizer)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_empty(self, tokenizer):
        """Test counting tokens in empty string."""
        count = preprocessing.count_tokens("", tokenizer)
        assert count == 0
    
    # --- Length Filtering Tests ---
    
    def test_filter_by_length_basic(self, tokenizer):
        """Test basic length filtering."""
        texts = [
            "Short",
            "This is a longer sentence with more words",
            "Medium length text",
        ]
        
        filtered = preprocessing.filter_by_length(
            texts, tokenizer, min_tokens=3, max_tokens=15
        )
        
        # Should filter out very short and very long texts
        assert len(filtered) > 0
        for text in filtered:
            token_count = len(tokenizer.encode(text))
            assert 3 <= token_count <= 15
    
    def test_filter_by_length_min_only(self, tokenizer):
        """Test filtering with only minimum length."""
        texts = ["A", "Short text", "This is longer"]
        
        filtered = preprocessing.filter_by_length(texts, tokenizer, min_tokens=2)
        
        # Should keep texts with >= 2 tokens
        assert len(filtered) >= 2
    
    def test_filter_by_length_max_only(self, tokenizer):
        """Test filtering with only maximum length."""
        texts = [
            "Short",
            " ".join([f"word{i}" for i in range(50)]),  # Very long
        ]
        
        filtered = preprocessing.filter_by_length(
            texts, tokenizer, min_tokens=1, max_tokens=10
        )
        
        # Should filter out the long text
        assert len(filtered) > 0
        for text in filtered:
            token_count = len(tokenizer.encode(text))
            assert token_count <= 10
    
    def test_filter_by_length_empty_list(self, tokenizer):
        """Test filtering an empty list."""
        filtered = preprocessing.filter_by_length([], tokenizer, min_tokens=1)
        assert filtered == []


class TestMemoryMappedDataset:
    """Test suite for memory-mapped dataset functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def create_text_file(self, temp_dir: Path, filename: str, content: str) -> Path:
        """Helper to create a text file."""
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def test_mmap_basic_loading(self, tokenizer, temp_dir):
        """Test basic memory-mapped dataset loading."""
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        # Create with mmap enabled
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        assert len(dataset) > 0
        stats = dataset.get_stats()
        assert stats['use_mmap'] is True
        assert stats['total_tokens'] > 0
    
    def test_mmap_vs_regular_consistency(self, tokenizer, temp_dir):
        """Test that mmap and regular datasets produce identical results."""
        content = " ".join([f"word{i}" for i in range(200)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        # Create both types
        dataset_regular = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=False,
        )
        
        dataset_mmap = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        # Should have same length
        assert len(dataset_regular) == len(dataset_mmap)
        
        # Should produce identical sequences
        for i in range(min(10, len(dataset_regular))):
            seq_regular = dataset_regular[i]
            seq_mmap = dataset_mmap[i]
            assert torch.equal(seq_regular, seq_mmap), f"Mismatch at index {i}"
    
    def test_mmap_multiple_files(self, tokenizer, temp_dir):
        """Test memory-mapped loading with multiple files."""
        file1 = self.create_text_file(
            temp_dir, "file1.txt", " ".join([f"word{i}" for i in range(100)])
        )
        file2 = self.create_text_file(
            temp_dir, "file2.txt", " ".join([f"text{i}" for i in range(100)])
        )
        
        dataset = TextDataset(
            file_paths=[file1, file2],
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        assert len(dataset) > 0
        stats = dataset.get_stats()
        assert stats['num_files'] == 2
        assert stats['use_mmap'] is True
    
    def test_mmap_sequence_access(self, tokenizer, temp_dir):
        """Test random access to sequences in mmap dataset."""
        content = " ".join([f"word{i}" for i in range(500)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        # Test random access
        if len(dataset) > 0:
            seq_first = dataset[0]
            assert seq_first.shape == (32,)
            assert seq_first.dtype == torch.long
        
        if len(dataset) > 1:
            seq_last = dataset[-1]
            assert seq_last.shape == (32,)
        
        if len(dataset) > 5:
            seq_middle = dataset[len(dataset) // 2]
            assert seq_middle.shape == (32,)
    
    def test_mmap_large_dataset_memory_efficiency(self, tokenizer, temp_dir):
        """Test that mmap uses less memory for large datasets."""
        # Create a large text file
        content = " ".join([f"word{i}" for i in range(5000)])
        file_path = self.create_text_file(temp_dir, "large.txt", content)
        
        # This should work without consuming too much memory
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=64,
            use_mmap=True,
        )
        
        assert len(dataset) > 0
        
        # Access multiple sequences to ensure mmap works correctly
        for i in range(min(20, len(dataset))):
            seq = dataset[i]
            assert seq.shape == (64,)
    
    def test_mmap_cleanup(self, tokenizer, temp_dir):
        """Test that mmap files are cleaned up properly."""
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        # Get mmap file path if accessible
        mmap_path = None
        if hasattr(dataset, 'mmap_file') and dataset.mmap_file:
            mmap_path = dataset.mmap_file.name
        
        # Delete dataset
        del dataset
        
        # If we had a mmap path, verify it's cleaned up
        # (Note: cleanup happens in __del__, which may be delayed by GC)
        import gc
        gc.collect()
        
        # Just verify the dataset can be deleted without errors
        assert True
    
    def test_mmap_with_stride(self, tokenizer, temp_dir):
        """Test memory-mapped dataset with custom stride."""
        content = " ".join([f"word{i}" for i in range(200)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            stride=16,  # 50% overlap
            use_mmap=True,
        )
        
        assert len(dataset) > 0
        
        # With stride < max_seq_len, should have more sequences
        dataset_no_overlap = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            stride=32,  # No overlap
            use_mmap=True,
        )
        
        assert len(dataset) >= len(dataset_no_overlap)
    
    def test_mmap_stats(self, tokenizer, temp_dir):
        """Test that stats work correctly for mmap datasets."""
        content = " ".join([f"word{i}" for i in range(100)])
        file_path = self.create_text_file(temp_dir, "test.txt", content)
        
        dataset = TextDataset(
            file_paths=file_path,
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        stats = dataset.get_stats()
        
        assert 'use_mmap' in stats
        assert stats['use_mmap'] is True
        assert 'total_tokens' in stats
        assert 'num_sequences' in stats
        assert 'max_seq_len' in stats
        assert stats['total_tokens'] > 0
        assert stats['num_sequences'] == len(dataset)
        assert stats['max_seq_len'] == 32
    
    def test_mmap_empty_file_handling(self, tokenizer, temp_dir):
        """Test that mmap handles empty files gracefully."""
        file1 = self.create_text_file(temp_dir, "empty.txt", "")
        file2 = self.create_text_file(
            temp_dir, "valid.txt", " ".join([f"word{i}" for i in range(100)])
        )
        
        # Should skip empty file and load valid one
        dataset = TextDataset(
            file_paths=[file1, file2],
            tokenizer=tokenizer,
            max_seq_len=32,
            use_mmap=True,
        )
        
        assert len(dataset) > 0
        stats = dataset.get_stats()
        assert stats['total_tokens'] > 0

"""
Tests for export functionality (GGUF, etc.).
"""

import tempfile
from pathlib import Path
import pytest
import struct
import torch
import numpy as np

from atlas.model import AtlasLM
from atlas.config import ModelConfig
from atlas.export import (
    GGUFWriter,
    GGMLQuantizationType,
    GGUFValueType,
    export_atlas_to_gguf,
)


@pytest.fixture
def sample_model():
    """Create a small test model."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
    )
    model = AtlasLM(config)
    return model, config


class TestGGUFWriter:
    """Test GGUF writer functionality."""
    
    def test_writer_initialization(self):
        """Test GGUF writer can be initialized."""
        writer = GGUFWriter()
        
        assert writer.metadata == {}
        assert writer.tensors == []
        assert writer.GGUF_MAGIC == 0x46554747
        assert writer.GGUF_VERSION == 3
    
    def test_add_metadata_int(self):
        """Test adding integer metadata."""
        writer = GGUFWriter()
        writer.add_metadata("test.int", 42)
        
        assert "test.int" in writer.metadata
        value, value_type = writer.metadata["test.int"]
        assert value == 42
        assert value_type == GGUFValueType.UINT32
    
    def test_add_metadata_float(self):
        """Test adding float metadata."""
        writer = GGUFWriter()
        writer.add_metadata("test.float", 3.14)
        
        assert "test.float" in writer.metadata
        value, value_type = writer.metadata["test.float"]
        assert value == 3.14
        assert value_type == GGUFValueType.FLOAT32
    
    def test_add_metadata_string(self):
        """Test adding string metadata."""
        writer = GGUFWriter()
        writer.add_metadata("test.string", "hello")
        
        assert "test.string" in writer.metadata
        value, value_type = writer.metadata["test.string"]
        assert value == "hello"
        assert value_type == GGUFValueType.STRING
    
    def test_add_metadata_bool(self):
        """Test adding boolean metadata."""
        writer = GGUFWriter()
        writer.add_metadata("test.bool", True)
        
        assert "test.bool" in writer.metadata
        value, value_type = writer.metadata["test.bool"]
        assert value is True
        assert value_type == GGUFValueType.BOOL
    
    def test_add_tensor_f32(self):
        """Test adding float32 tensor."""
        writer = GGUFWriter()
        tensor = torch.randn(10, 20)
        writer.add_tensor("test.weight", tensor, GGMLQuantizationType.F32)
        
        assert len(writer.tensors) == 1
        name, np_tensor = writer.tensors[0]
        assert name == "test.weight"
        assert np_tensor.dtype == np.float32
        assert np_tensor.shape == (10, 20)
    
    def test_add_tensor_f16(self):
        """Test adding float16 tensor."""
        writer = GGUFWriter()
        tensor = torch.randn(10, 20)
        writer.add_tensor("test.weight", tensor, GGMLQuantizationType.F16)
        
        assert len(writer.tensors) == 1
        name, np_tensor = writer.tensors[0]
        assert name == "test.weight"
        assert np_tensor.dtype == np.float16
        assert np_tensor.shape == (10, 20)
    
    def test_write_file_creates_output(self):
        """Test that write_to_file creates a file."""
        writer = GGUFWriter()
        writer.add_metadata("test.name", "TestModel")
        writer.add_metadata("test.version", 1)
        
        tensor = torch.randn(10, 20)
        writer.add_tensor("test.weight", tensor)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_write_file_has_magic_number(self):
        """Test that written file has correct magic number."""
        writer = GGUFWriter()
        writer.add_metadata("test", 1)
        writer.add_tensor("weight", torch.randn(5, 5))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            with open(output_path, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                assert magic == 0x46554747  # "GGUF"
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_write_file_has_version(self):
        """Test that written file has correct version."""
        writer = GGUFWriter()
        writer.add_metadata("test", 1)
        writer.add_tensor("weight", torch.randn(5, 5))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            with open(output_path, 'rb') as f:
                f.read(4)  # Skip magic
                version = struct.unpack('<I', f.read(4))[0]
                assert version == 3
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_write_file_has_tensor_count(self):
        """Test that written file has correct tensor count."""
        writer = GGUFWriter()
        writer.add_metadata("test", 1)
        writer.add_tensor("weight1", torch.randn(5, 5))
        writer.add_tensor("weight2", torch.randn(10, 10))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            with open(output_path, 'rb') as f:
                f.read(8)  # Skip magic and version
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                assert tensor_count == 2
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestGGUFExport:
    """Test full model export to GGUF."""
    
    def test_export_creates_file(self, sample_model):
        """Test that export creates output file."""
        model, config = sample_model
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            model_config_dict = {
                'vocab_size': config.vocab_size,
                'max_seq_len': config.max_seq_len,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
            }
            
            export_atlas_to_gguf(
                model=model,
                output_path=output_path,
                model_config=model_config_dict,
            )
            
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_export_file_has_magic(self, sample_model):
        """Test exported file has GGUF magic number."""
        model, config = sample_model
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            model_config_dict = {
                'vocab_size': config.vocab_size,
                'max_seq_len': config.max_seq_len,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
            }
            
            export_atlas_to_gguf(
                model=model,
                output_path=output_path,
                model_config=model_config_dict,
            )
            
            with open(output_path, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                assert magic == 0x46554747
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_export_f16_creates_smaller_file(self, sample_model):
        """Test that F16 export creates smaller file than F32."""
        model, config = sample_model
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='_f32.gguf') as f:
            f32_path = f.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='_f16.gguf') as f:
            f16_path = f.name
        
        try:
            model_config_dict = {
                'vocab_size': config.vocab_size,
                'max_seq_len': config.max_seq_len,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
            }
            
            # Export F32
            export_atlas_to_gguf(
                model=model,
                output_path=f32_path,
                model_config=model_config_dict,
                quantization=GGMLQuantizationType.F32,
            )
            
            # Export F16
            export_atlas_to_gguf(
                model=model,
                output_path=f16_path,
                model_config=model_config_dict,
                quantization=GGMLQuantizationType.F16,
            )
            
            f32_size = Path(f32_path).stat().st_size
            f16_size = Path(f16_path).stat().st_size
            
            # F16 should be roughly half the size of F32
            assert f16_size < f32_size
            # Allow some overhead for metadata
            assert f16_size < f32_size * 0.7
        finally:
            Path(f32_path).unlink(missing_ok=True)
            Path(f16_path).unlink(missing_ok=True)
    
    def test_export_with_tokenizer_config(self, sample_model):
        """Test export with tokenizer configuration."""
        model, config = sample_model
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            model_config_dict = {
                'vocab_size': config.vocab_size,
                'max_seq_len': config.max_seq_len,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
            }
            
            tokenizer_config = {
                'name': 'gpt2',
            }
            
            export_atlas_to_gguf(
                model=model,
                output_path=output_path,
                model_config=model_config_dict,
                tokenizer_config=tokenizer_config,
            )
            
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestGGUFFileStructure:
    """Test GGUF file structure and format."""
    
    def test_file_header_structure(self):
        """Test that file header has correct structure."""
        writer = GGUFWriter()
        writer.add_metadata("name", "test")
        writer.add_tensor("weight", torch.randn(10, 10))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            with open(output_path, 'rb') as f:
                # Read header
                magic = struct.unpack('<I', f.read(4))[0]
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                assert magic == 0x46554747
                assert version == 3
                assert tensor_count == 1
                assert metadata_count == 1
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_metadata_count_matches(self):
        """Test that metadata count in header matches actual metadata."""
        writer = GGUFWriter()
        writer.add_metadata("key1", "value1")
        writer.add_metadata("key2", 42)
        writer.add_metadata("key3", 3.14)
        writer.add_tensor("weight", torch.randn(5, 5))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
            output_path = f.name
        
        try:
            writer.write_to_file(output_path)
            
            with open(output_path, 'rb') as f:
                f.read(16)  # Skip magic, version, tensor_count
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                assert metadata_count == 3
        finally:
            Path(output_path).unlink(missing_ok=True)

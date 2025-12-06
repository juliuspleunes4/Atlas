"""
GGUF (GPT-Generated Unified Format) export for Atlas models.

GGUF is a file format for storing models for inference with llama.cpp.
This module handles conversion from PyTorch to GGUF format.

GGUF Format Overview:
- Magic number: "GGUF" (0x46554747)
- Version: uint32
- Tensor count: uint64
- Metadata count: uint64
- Metadata key-value pairs
- Tensor infos (name, dimensions, type, offset)
- Alignment padding
- Tensor data

Reference: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf.py
"""

from typing import Any, Dict, List, Optional, BinaryIO
from dataclasses import dataclass
from enum import IntEnum
import struct
import numpy as np
import torch


class GGMLQuantizationType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in GGUF format."""
    name: str
    shape: List[int]
    dtype: GGMLQuantizationType
    offset: int


class GGUFWriter:
    """
    Writer for GGUF format files.
    
    This class handles serialization of model metadata and tensors
    to the GGUF format for use with llama.cpp and compatible tools.
    """
    
    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3
    ALIGNMENT = 32  # Byte alignment for tensor data
    
    def __init__(self):
        """Initialize GGUF writer."""
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[tuple[str, np.ndarray]] = []
    
    def add_metadata(self, key: str, value: Any, value_type: Optional[GGUFValueType] = None):
        """
        Add metadata key-value pair.
        
        Args:
            key: Metadata key
            value: Metadata value
            value_type: Optional explicit value type
        """
        if value_type is None:
            # Infer type from value
            if isinstance(value, bool):
                value_type = GGUFValueType.BOOL
            elif isinstance(value, int):
                value_type = GGUFValueType.UINT32
            elif isinstance(value, float):
                value_type = GGUFValueType.FLOAT32
            elif isinstance(value, str):
                value_type = GGUFValueType.STRING
            elif isinstance(value, (list, tuple)):
                value_type = GGUFValueType.ARRAY
            else:
                raise ValueError(f"Cannot infer GGUF type for value: {type(value)}")
        
        self.metadata[key] = (value, value_type)
    
    def add_tensor(self, name: str, tensor: torch.Tensor, quantization: GGMLQuantizationType = GGMLQuantizationType.F32):
        """
        Add tensor for export.
        
        Args:
            name: Tensor name (used as key in GGUF file)
            tensor: PyTorch tensor
            quantization: Quantization type (default: F32)
        """
        # Convert to numpy and ensure correct dtype
        if quantization == GGMLQuantizationType.F32:
            np_tensor = tensor.detach().cpu().float().numpy()
        elif quantization == GGMLQuantizationType.F16:
            np_tensor = tensor.detach().cpu().half().numpy()
        else:
            raise NotImplementedError(f"Quantization type {quantization} not yet implemented")
        
        self.tensors.append((name, np_tensor))
    
    def write_to_file(self, file_path: str):
        """
        Write GGUF file to disk.
        
        Args:
            file_path: Output file path
        """
        with open(file_path, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write metadata
            self._write_metadata(f)
            
            # Write tensor infos
            tensor_infos = self._write_tensor_infos(f)
            
            # Align to ALIGNMENT bytes
            self._write_alignment(f)
            
            # Write tensor data
            self._write_tensor_data(f, tensor_infos)
    
    def _write_header(self, f: BinaryIO):
        """Write GGUF header."""
        f.write(struct.pack('<I', self.GGUF_MAGIC))  # Magic
        f.write(struct.pack('<I', self.GGUF_VERSION))  # Version
        f.write(struct.pack('<Q', len(self.tensors)))  # Tensor count
        f.write(struct.pack('<Q', len(self.metadata)))  # Metadata count
    
    def _write_metadata(self, f: BinaryIO):
        """Write metadata key-value pairs."""
        for key, (value, value_type) in self.metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Write value type
            f.write(struct.pack('<I', value_type))
            
            # Write value
            self._write_value(f, value, value_type)
    
    def _write_value(self, f: BinaryIO, value: Any, value_type: GGUFValueType):
        """Write a single metadata value."""
        if value_type == GGUFValueType.UINT8:
            f.write(struct.pack('<B', value))
        elif value_type == GGUFValueType.INT8:
            f.write(struct.pack('<b', value))
        elif value_type == GGUFValueType.UINT16:
            f.write(struct.pack('<H', value))
        elif value_type == GGUFValueType.INT16:
            f.write(struct.pack('<h', value))
        elif value_type == GGUFValueType.UINT32:
            f.write(struct.pack('<I', value))
        elif value_type == GGUFValueType.INT32:
            f.write(struct.pack('<i', value))
        elif value_type == GGUFValueType.UINT64:
            f.write(struct.pack('<Q', value))
        elif value_type == GGUFValueType.INT64:
            f.write(struct.pack('<q', value))
        elif value_type == GGUFValueType.FLOAT32:
            f.write(struct.pack('<f', value))
        elif value_type == GGUFValueType.FLOAT64:
            f.write(struct.pack('<d', value))
        elif value_type == GGUFValueType.BOOL:
            f.write(struct.pack('<B', 1 if value else 0))
        elif value_type == GGUFValueType.STRING:
            str_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(str_bytes)))
            f.write(str_bytes)
        elif value_type == GGUFValueType.ARRAY:
            # Arrays have type, then count, then elements
            if len(value) == 0:
                raise ValueError("Cannot write empty array to GGUF")
            
            # Infer array element type
            elem = value[0]
            if isinstance(elem, int):
                elem_type = GGUFValueType.UINT32
            elif isinstance(elem, float):
                elem_type = GGUFValueType.FLOAT32
            elif isinstance(elem, str):
                elem_type = GGUFValueType.STRING
            else:
                raise ValueError(f"Unsupported array element type: {type(elem)}")
            
            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(value)))
            
            for elem in value:
                self._write_value(f, elem, elem_type)
        else:
            raise ValueError(f"Unsupported value type: {value_type}")
    
    def _write_tensor_infos(self, f: BinaryIO) -> List[GGUFTensorInfo]:
        """Write tensor information and return list of tensor infos."""
        tensor_infos = []
        current_offset = 0
        
        for name, np_tensor in self.tensors:
            # Determine GGML type
            if np_tensor.dtype == np.float32:
                ggml_type = GGMLQuantizationType.F32
            elif np_tensor.dtype == np.float16:
                ggml_type = GGMLQuantizationType.F16
            else:
                raise ValueError(f"Unsupported tensor dtype: {np_tensor.dtype}")
            
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write dimensions
            n_dims = len(np_tensor.shape)
            f.write(struct.pack('<I', n_dims))
            for dim in np_tensor.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write type
            f.write(struct.pack('<I', ggml_type))
            
            # Write offset
            f.write(struct.pack('<Q', current_offset))
            
            # Track info
            tensor_info = GGUFTensorInfo(
                name=name,
                shape=list(np_tensor.shape),
                dtype=ggml_type,
                offset=current_offset
            )
            tensor_infos.append(tensor_info)
            
            # Update offset
            current_offset += np_tensor.nbytes
        
        return tensor_infos
    
    def _write_alignment(self, f: BinaryIO):
        """Write alignment padding."""
        current_pos = f.tell()
        padding = (self.ALIGNMENT - (current_pos % self.ALIGNMENT)) % self.ALIGNMENT
        f.write(b'\x00' * padding)
    
    def _write_tensor_data(self, f: BinaryIO, tensor_infos: List[GGUFTensorInfo]):
        """Write actual tensor data."""
        for (name, np_tensor), tensor_info in zip(self.tensors, tensor_infos):
            # Write tensor data in row-major order (C order)
            np_tensor.tofile(f)


def export_atlas_to_gguf(
    model: torch.nn.Module,
    output_path: str,
    model_config: Dict[str, Any],
    tokenizer_config: Optional[Dict[str, Any]] = None,
    quantization: GGMLQuantizationType = GGMLQuantizationType.F32,
):
    """
    Export Atlas model to GGUF format.
    
    Args:
        model: PyTorch model to export
        output_path: Output GGUF file path
        model_config: Model configuration dictionary
        tokenizer_config: Optional tokenizer configuration
        quantization: Quantization type (default: F32)
    """
    writer = GGUFWriter()
    
    # Add metadata
    writer.add_metadata("general.architecture", "atlas")
    writer.add_metadata("general.name", "Atlas LLM")
    
    # Model architecture metadata
    writer.add_metadata("atlas.vocab_size", model_config['vocab_size'])
    writer.add_metadata("atlas.context_length", model_config['max_seq_len'])
    writer.add_metadata("atlas.embedding_length", model_config['hidden_size'])
    writer.add_metadata("atlas.block_count", model_config['num_layers'])
    writer.add_metadata("atlas.attention.head_count", model_config['num_heads'])
    
    # Optional: tokenizer info
    if tokenizer_config:
        writer.add_metadata("tokenizer.ggml.model", tokenizer_config.get('name', 'gpt2'))
    
    # Add tensors with mapping from Atlas names to GGUF names
    state_dict = model.state_dict()
    
    # Mapping from Atlas tensor names to GGUF tensor names
    tensor_map = {
        # Token embeddings
        'embeddings.token_embedding.embedding.weight': 'token_embd.weight',
        # Positional embeddings
        'embeddings.positional_embedding.embedding.weight': 'position_embd.weight',
        # Output head
        'lm_head.weight': 'output.weight',
    }
    
    # Add transformer blocks
    for i in range(model_config['num_layers']):
        tensor_map.update({
            f'blocks.{i}.ln1.weight': f'blk.{i}.attn_norm.weight',
            f'blocks.{i}.ln1.bias': f'blk.{i}.attn_norm.bias',
            f'blocks.{i}.attention.qkv_proj.weight': f'blk.{i}.attn_qkv.weight',
            f'blocks.{i}.attention.qkv_proj.bias': f'blk.{i}.attn_qkv.bias',
            f'blocks.{i}.attention.out_proj.weight': f'blk.{i}.attn_output.weight',
            f'blocks.{i}.attention.out_proj.bias': f'blk.{i}.attn_output.bias',
            f'blocks.{i}.ln2.weight': f'blk.{i}.ffn_norm.weight',
            f'blocks.{i}.ln2.bias': f'blk.{i}.ffn_norm.bias',
            f'blocks.{i}.mlp.fc1.weight': f'blk.{i}.ffn_up.weight',
            f'blocks.{i}.mlp.fc1.bias': f'blk.{i}.ffn_up.bias',
            f'blocks.{i}.mlp.fc2.weight': f'blk.{i}.ffn_down.weight',
            f'blocks.{i}.mlp.fc2.bias': f'blk.{i}.ffn_down.bias',
        })
    
    # Export tensors
    for atlas_name, gguf_name in tensor_map.items():
        if atlas_name in state_dict:
            tensor = state_dict[atlas_name]
            writer.add_tensor(gguf_name, tensor, quantization)
        else:
            print(f"Warning: Tensor {atlas_name} not found in state dict")
    
    # Write to file
    writer.write_to_file(output_path)
    print(f"Exported model to {output_path}")
    print(f"  Tensors: {len(writer.tensors)}")
    print(f"  Quantization: {quantization.name}")

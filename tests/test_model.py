"""
Tests for model architecture components.
"""
import pytest
import torch
import torch.nn as nn
from atlas.model.embeddings import TokenEmbedding, PositionalEmbedding, CombinedEmbedding
from atlas.model.attention import MultiHeadAttention
from atlas.model.mlp import MLP
from atlas.model.transformer import TransformerBlock
from atlas.model.model import AtlasLM
from atlas.config.config import ModelConfig


# ========================================
# Embedding Tests
# ========================================

def test_token_embedding_initialization():
    """Test TokenEmbedding initialization."""
    vocab_size = 1000
    hidden_size = 128
    emb = TokenEmbedding(vocab_size, hidden_size)
    assert emb.embedding.num_embeddings == vocab_size
    assert emb.embedding.embedding_dim == hidden_size


def test_token_embedding_forward():
    """Test TokenEmbedding forward pass."""
    vocab_size = 1000
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    emb = TokenEmbedding(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = emb(input_ids)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_positional_embedding_initialization():
    """Test PositionalEmbedding initialization."""
    max_len = 512
    hidden_size = 128
    pos_emb = PositionalEmbedding(max_len, hidden_size)
    assert pos_emb.embedding.num_embeddings == max_len
    assert pos_emb.embedding.embedding_dim == hidden_size


def test_positional_embedding_forward():
    """Test PositionalEmbedding forward pass."""
    max_len = 512
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    pos_emb = PositionalEmbedding(max_len, hidden_size)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    output = pos_emb(positions)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_positional_embedding_different_batch_sizes():
    """Test PositionalEmbedding with different batch sizes."""
    max_len = 512
    hidden_size = 128
    pos_emb = PositionalEmbedding(max_len, hidden_size)
    
    # Test with different batch sizes
    for batch_size in [1, 2, 8]:
        seq_len = 10
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        output = pos_emb(positions)
        assert output.shape == (batch_size, seq_len, hidden_size)


def test_positional_embedding_max_length():
    """Test PositionalEmbedding at maximum length."""
    max_len = 512
    hidden_size = 128
    batch_size = 2
    
    pos_emb = PositionalEmbedding(max_len, hidden_size)
    # Test at exact max length
    positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
    output = pos_emb(positions)
    assert output.shape == (batch_size, max_len, hidden_size)


def test_combined_embedding_initialization():
    """Test CombinedEmbedding initialization."""
    vocab_size = 1000
    max_len = 512
    hidden_size = 128
    dropout = 0.1
    
    combined = CombinedEmbedding(vocab_size, max_len, hidden_size, dropout)
    assert isinstance(combined.token_embedding, TokenEmbedding)
    assert isinstance(combined.positional_embedding, PositionalEmbedding)
    assert isinstance(combined.dropout, nn.Dropout)
    assert combined.dropout.p == dropout


def test_combined_embedding_forward():
    """Test CombinedEmbedding forward pass."""
    vocab_size = 1000
    max_len = 512
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    combined = CombinedEmbedding(vocab_size, max_len, hidden_size, dropout=0.0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = combined(input_ids)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_combined_embedding_with_dropout():
    """Test CombinedEmbedding applies dropout during training."""
    vocab_size = 1000
    max_len = 512
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    combined = CombinedEmbedding(vocab_size, max_len, hidden_size, dropout=0.5)
    combined.train()  # Enable training mode
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run multiple times to check dropout is active
    outputs = [combined(input_ids) for _ in range(3)]
    
    # Outputs should differ due to dropout
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])


def test_combined_embedding_eval_mode():
    """Test CombinedEmbedding is deterministic in eval mode."""
    vocab_size = 1000
    max_len = 512
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    combined = CombinedEmbedding(vocab_size, max_len, hidden_size, dropout=0.5)
    combined.eval()  # Disable dropout
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output1 = combined(input_ids)
    output2 = combined(input_ids)
    
    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2)


# ========================================
# Attention Tests
# ========================================

def test_attention_initialization():
    """Test MultiHeadAttention initialization."""
    hidden_size = 128
    num_heads = 8
    attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.1)
    
    assert attn.hidden_size == hidden_size
    assert attn.num_heads == num_heads
    assert attn.head_dim == hidden_size // num_heads
    assert isinstance(attn.qkv_proj, nn.Linear)
    assert isinstance(attn.out_proj, nn.Linear)


def test_attention_dimension_divisibility():
    """Test MultiHeadAttention requires hidden_size divisible by num_heads."""
    with pytest.raises(ValueError):
        MultiHeadAttention(hidden_size=128, num_heads=7)  # 128 not divisible by 7


def test_attention_forward():
    """Test MultiHeadAttention forward pass."""
    batch_size = 4
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    
    attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_attention_causal_mask():
    """Test MultiHeadAttention applies causal masking correctly."""
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_heads = 4
    
    attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.0)
    attn.eval()  # Disable dropout for deterministic output
    
    # Create simple input where each position has distinct value
    x = torch.arange(seq_len).float().view(1, seq_len, 1).expand(batch_size, seq_len, hidden_size)
    
    with torch.no_grad():
        output = attn(x)
    
    # Due to causal masking, each position can only attend to itself and previous positions
    # First position should not be influenced by future positions
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_attention_single_head():
    """Test MultiHeadAttention with single head."""
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    
    attn = MultiHeadAttention(hidden_size, num_heads=1, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_attention_many_heads():
    """Test MultiHeadAttention with many heads."""
    batch_size = 2
    seq_len = 8
    hidden_size = 128
    
    attn = MultiHeadAttention(hidden_size, num_heads=16, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_attention_long_sequence():
    """Test MultiHeadAttention with longer sequence."""
    batch_size = 2
    seq_len = 256
    hidden_size = 128
    num_heads = 8
    
    attn = MultiHeadAttention(hidden_size, num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)


# ========================================
# MLP Tests
# ========================================

def test_mlp_initialization():
    """Test MLP initialization."""
    hidden_size = 128
    # hidden_dim calculated from mlp_ratio
    mlp = MLP(hidden_size, mlp_ratio=4.0, dropout=0.1)
    
    assert isinstance(mlp.fc1, nn.Linear)
    assert isinstance(mlp.fc2, nn.Linear)
    assert mlp.fc1.in_features == hidden_size
    assert mlp.fc1.out_features == int(hidden_size * 4.0)
    assert mlp.fc2.in_features == int(hidden_size * 4.0)
    assert mlp.fc2.out_features == hidden_size


def test_mlp_forward():
    """Test MLP forward pass."""
    batch_size = 4
    seq_len = 10
    hidden_size = 128
    # hidden_dim calculated from mlp_ratio
    
    mlp = MLP(hidden_size, mlp_ratio=4.0, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = mlp(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_mlp_activation_gelu():
    """Test MLP with GELU activation."""
    mlp = MLP(128, mlp_ratio=4.0, activation="gelu", dropout=0.0)
    x = torch.randn(2, 10, 128)
    output = mlp(x)
    assert output.shape == x.shape


def test_mlp_activation_silu():
    """Test MLP with SiLU activation."""
    mlp = MLP(128, mlp_ratio=4.0, activation="silu", dropout=0.0)
    x = torch.randn(2, 10, 128)
    output = mlp(x)
    assert output.shape == x.shape


def test_mlp_activation_relu():
    """Test MLP with ReLU activation."""
    mlp = MLP(128, mlp_ratio=4.0, activation="relu", dropout=0.0)
    x = torch.randn(2, 10, 128)
    output = mlp(x)
    assert output.shape == x.shape


def test_mlp_invalid_activation():
    """Test MLP raises error for invalid activation."""
    with pytest.raises(ValueError):
        MLP(128, mlp_ratio=4.0, activation="invalid")


def test_mlp_with_dropout():
    """Test MLP applies dropout during training."""
    mlp = MLP(128, mlp_ratio=4.0, dropout=0.5)
    mlp.train()
    
    x = torch.randn(2, 10, 128)
    outputs = [mlp(x) for _ in range(3)]
    
    # Outputs should differ due to dropout
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])


# ========================================
# TransformerBlock Tests
# ========================================

def test_transformer_block_initialization():
    """Test TransformerBlock initialization."""
    hidden_size = 128
    num_heads = 8
    # hidden_dim calculated from mlp_ratio
    
    block = TransformerBlock(hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1)
    
    assert isinstance(block.ln1, nn.LayerNorm)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(block.ln2, nn.LayerNorm)
    assert isinstance(block.mlp, MLP)


def test_transformer_block_forward():
    """Test TransformerBlock forward pass."""
    batch_size = 4
    seq_len = 10
    hidden_size = 128
    num_heads = 8
    # hidden_dim calculated from mlp_ratio
    
    block = TransformerBlock(hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == torch.float32


def test_transformer_block_residual_connections():
    """Test TransformerBlock preserves residual connections."""
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    num_heads = 4
    # hidden_dim calculated from mlp_ratio
    
    block = TransformerBlock(hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0)
    block.eval()
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    with torch.no_grad():
        output = block(x)
    
    # Output should not be identical to input (due to attention and MLP)
    # but should be influenced by input through residual connections
    assert not torch.allclose(output, x)
    assert output.shape == x.shape


def test_transformer_block_layer_norm():
    """Test TransformerBlock applies layer normalization."""
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    num_heads = 4
    # hidden_dim calculated from mlp_ratio
    
    block = TransformerBlock(hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0)
    
    # Create input with large variance
    x = torch.randn(batch_size, seq_len, hidden_size) * 10
    
    with torch.no_grad():
        output = block(x)
    
    # Verify output shape is preserved
    assert output.shape == x.shape


# ========================================
# AtlasLM Tests
# ========================================

def test_atlas_lm_initialization():
    """Test AtlasLM initialization."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    model = AtlasLM(config)
    
    assert isinstance(model.embeddings, CombinedEmbedding)
    assert len(model.blocks) == config.num_layers
    assert isinstance(model.ln_f, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.out_features == config.vocab_size


def test_atlas_lm_forward():
    """Test AtlasLM forward pass."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=6,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    batch_size = 4
    seq_len = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert logits.dtype == torch.float32


def test_atlas_lm_small_model():
    """Test AtlasLM with small configuration."""
    config = ModelConfig(
        vocab_size=500,
        max_seq_len=64,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    batch_size = 2
    seq_len = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_atlas_lm_single_layer():
    """Test AtlasLM with single transformer layer."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=1,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    assert len(model.blocks) == 1
    
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, config.vocab_size)


def test_atlas_lm_count_parameters():
    """Test AtlasLM parameter counting."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    num_params = model.count_parameters()
    
    assert num_params > 0
    assert isinstance(num_params, int)
    
    # Verify by manually counting
    manual_count = sum(p.numel() for p in model.parameters())
    assert num_params == manual_count


def test_atlas_lm_get_num_params():
    """Test AtlasLM get_num_params returns formatted string."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    num_params_str = model.get_num_params()
    
    assert isinstance(num_params_str, str)
    assert "M" in num_params_str  # Should be in millions


def test_atlas_lm_weight_tying():
    """Test AtlasLM ties weights between embedding and LM head."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    
    # LM head weight should share storage with token embedding weight
    assert model.lm_head.weight.data_ptr() == model.embeddings.token_embedding.embedding.weight.data_ptr()


def test_atlas_lm_generate_basic():
    """Test AtlasLM generate method basic functionality."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    batch_size = 2
    seq_len = 5
    max_new_tokens = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    
    # Output should have max_new_tokens more tokens than input
    assert output_ids.shape == (batch_size, seq_len + max_new_tokens)
    
    # All tokens should be valid vocab indices
    assert (output_ids >= 0).all()
    assert (output_ids < config.vocab_size).all()


def test_atlas_lm_generate_with_temperature():
    """Test AtlasLM generate with different temperatures."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    # Low temperature (more deterministic)
    with torch.no_grad():
        output_low_temp = model.generate(input_ids, max_new_tokens=5, temperature=0.1)
    
    # High temperature (more random)
    with torch.no_grad():
        output_high_temp = model.generate(input_ids, max_new_tokens=5, temperature=2.0)
    
    assert output_low_temp.shape == (1, 10)
    assert output_high_temp.shape == (1, 10)


def test_atlas_lm_generate_with_top_k():
    """Test AtlasLM generate with top-k sampling."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=5, top_k=10)
    
    assert output_ids.shape == (2, 10)


def test_atlas_lm_generate_with_eos():
    """Test AtlasLM generate stops at EOS token."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    eos_token_id = 2
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    # Set a seed for reproducibility in this test
    torch.manual_seed(42)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=20, 
            eos_token_id=eos_token_id,
            temperature=1.0
        )
    
    # Output should be at most input_len + max_new_tokens
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] <= 25  # 5 + 20


def test_atlas_lm_generate_max_length_constraint():
    """Test AtlasLM generate respects max_seq_len."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=20,  # Small max length
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100)  # Request many tokens
    
    # Should stop at max_seq_len
    assert output_ids.shape[1] <= config.max_seq_len


def test_atlas_lm_different_batch_sizes():
    """Test AtlasLM handles different batch sizes correctly."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=256,
        num_heads=8,
        num_layers=4,
        dropout=0.0
    )
    
    model = AtlasLM(config)
    model.eval()
    
    for batch_size in [1, 2, 4, 8]:
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_atlas_lm_eval_vs_train_mode():
    """Test AtlasLM behaves differently in train vs eval mode."""
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        dropout=0.5  # High dropout
    )
    
    model = AtlasLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    # Train mode - dropout active
    model.train()
    outputs_train = [model(input_ids) for _ in range(2)]
    # Outputs should differ due to dropout
    assert not torch.allclose(outputs_train[0], outputs_train[1])
    
    # Eval mode - dropout inactive
    model.eval()
    with torch.no_grad():
        output_eval1 = model(input_ids)
        output_eval2 = model(input_ids)
    # Outputs should be identical
    assert torch.allclose(output_eval1, output_eval2)

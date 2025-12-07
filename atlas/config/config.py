"""
Configuration schemas for Atlas.

This module defines dataclasses for all configuration aspects of Atlas,
including model architecture, training parameters, data settings, and more.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Architecture parameters
    num_layers: int = 6
    hidden_size: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0  # MLP hidden dim = hidden_size * mlp_ratio
    vocab_size: int = 50257  # Default GPT-2 vocab size
    max_seq_len: int = 1024
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Architecture choices
    use_bias: bool = True
    norm_eps: float = 1e-5
    tie_weights: bool = True  # Tie input/output embeddings
    
    # Activation function
    activation: str = "gelu"  # gelu, silu, relu
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Trade compute for memory
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.activation not in ["gelu", "silu", "relu"]:
            raise ValueError(f"activation must be one of: gelu, silu, relu")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.attention_dropout < 0 or self.attention_dropout > 1:
            raise ValueError(
                f"attention_dropout must be in [0, 1], got {self.attention_dropout}"
            )
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_grad_norm: float = 1.0  # Alias for grad_clip (used in YAML)
    optimizer_type: str = "adamw"  # adamw, adamw8bit, sgd
    momentum: float = 0.9  # For SGD optimizer
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_schedule: str = "cosine"  # cosine, linear, constant
    scheduler_type: str = "cosine"  # Alias for lr_schedule (used in YAML)
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Trade compute for memory
    
    # Mixed precision
    use_amp: bool = False  # Automatic mixed precision
    
    # Evaluation
    eval_interval: int = 1000
    eval_steps: int = 100
    
    # Checkpointing
    keep_checkpoints: int = 3  # Number of step-based checkpoints to keep
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Sync aliases - prefer the non-default value
        # If scheduler_type was explicitly set (differs from default), use it
        if self.scheduler_type != "cosine" and self.lr_schedule == "cosine":
            self.lr_schedule = self.scheduler_type
        # Otherwise sync the other way
        elif self.lr_schedule != "cosine" and self.scheduler_type == "cosine":
            self.scheduler_type = self.lr_schedule
            
        # Sync grad clip
        if self.max_grad_norm != 1.0 and self.grad_clip == 1.0:
            self.grad_clip = self.max_grad_norm
        elif self.grad_clip != 1.0 and self.max_grad_norm == 1.0:
            self.max_grad_norm = self.grad_clip
            
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.lr_schedule not in ["cosine", "linear", "constant"]:
            raise ValueError(
                f"lr_schedule must be one of: cosine, linear, constant"
            )
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset paths
    train_data_path: Optional[str] = None  # TODO: Set actual path
    val_data_path: Optional[str] = None  # TODO: Set actual path
    
    # Tokenizer
    tokenizer_path: Optional[str] = None  # TODO: Set actual path
    
    # Data processing
    num_workers: int = 2  # Number of dataloader workers
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Sequence settings
    max_seq_len: int = 1024  # Maximum sequence length (used in YAML configs)
    sequence_length: int = 1024  # Alias for max_seq_len (used internally)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Sync max_seq_len and sequence_length
        if self.max_seq_len != 1024 and self.sequence_length == 1024:
            self.sequence_length = self.max_seq_len
        elif self.sequence_length != 1024 and self.max_seq_len == 1024:
            self.max_seq_len = self.sequence_length
            
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    
    # Directories
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Logging intervals
    log_interval: int = 100  # Log every N steps
    checkpoint_interval: int = 5000  # Save checkpoint every N steps
    
    # Checkpoint management
    keep_last_n_checkpoints: int = 5  # Keep only last N checkpoints
    save_optimizer_state: bool = True
    
    # Logging details
    log_gradients: bool = False
    log_model_structure: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )


@dataclass
class InferenceConfig:
    """Configuration for text generation and inference."""
    
    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Sampling strategy
    do_sample: bool = True  # If False, use greedy decoding
    
    # Special tokens
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass
class AtlasConfig:
    """Complete configuration for Atlas."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, or specific device like cuda:0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure sequence lengths match
        if self.data.sequence_length != self.model.max_seq_len:
            raise ValueError(
                f"data.sequence_length ({self.data.sequence_length}) must match "
                f"model.max_seq_len ({self.model.max_seq_len})"
            )

"""
Tests for training loop and optimization.
"""

import pytest
import torch
import math

from atlas.training.loss import (
    compute_lm_loss,
    compute_lm_loss_with_logits_shift,
    compute_perplexity,
)
from atlas.training.optimizer import (
    create_optimizer,
    create_scheduler,
    get_optimizer_state,
    clip_gradients,
)
from atlas.model import AtlasLM
from atlas.config import ModelConfig


class TestLossFunctions:
    """Test suite for loss functions."""
    
    def test_compute_lm_loss_basic(self):
        """Test basic loss computation."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_lm_loss(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
    
    def test_compute_lm_loss_reduction_mean(self):
        """Test loss with mean reduction."""
        logits = torch.randn(2, 10, 50)
        targets = torch.randint(0, 50, (2, 10))
        
        loss = compute_lm_loss(logits, targets, reduction='mean')
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)
    
    def test_compute_lm_loss_reduction_sum(self):
        """Test loss with sum reduction."""
        logits = torch.randn(2, 10, 50)
        targets = torch.randint(0, 50, (2, 10))
        
        loss_mean = compute_lm_loss(logits, targets, reduction='mean')
        loss_sum = compute_lm_loss(logits, targets, reduction='sum')
        
        assert loss_sum > loss_mean  # Sum should be larger than mean
    
    def test_compute_lm_loss_reduction_none(self):
        """Test loss with no reduction."""
        batch_size, seq_len = 2, 10
        logits = torch.randn(batch_size, seq_len, 50)
        targets = torch.randint(0, 50, (batch_size, seq_len))
        
        loss = compute_lm_loss(logits, targets, reduction='none')
        
        assert loss.shape == (batch_size, seq_len)
        assert not torch.any(torch.isnan(loss))
    
    def test_compute_lm_loss_ignore_index(self):
        """Test that ignore_index is properly handled."""
        logits = torch.randn(2, 10, 50)
        targets = torch.randint(0, 50, (2, 10))
        
        # Set some targets to ignore_index
        targets[:, 5:] = -100
        
        loss_with_ignore = compute_lm_loss(logits, targets, ignore_index=-100)
        
        # Loss should be computed only on non-ignored tokens
        assert not torch.isnan(loss_with_ignore)
        assert loss_with_ignore.item() > 0
    
    def test_compute_lm_loss_perfect_prediction(self):
        """Test loss with perfect predictions."""
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        # Create targets
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create perfect logits (large value for correct class)
        logits = torch.full((batch_size, seq_len, vocab_size), -100.0)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, targets[b, s]] = 100.0
        
        loss = compute_lm_loss(logits, targets)
        
        # Loss should be very small (close to 0)
        assert loss.item() < 0.01
    
    def test_compute_lm_loss_with_logits_shift(self):
        """Test loss computation with automatic shifting."""
        batch_size, seq_len, vocab_size = 2, 10, 50
        
        model_output = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_lm_loss_with_logits_shift(model_output, input_ids)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_compute_lm_loss_shift_shapes(self):
        """Test that shifting produces correct shapes."""
        batch_size, seq_len, vocab_size = 2, 10, 50
        
        model_output = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Manually shift
        shift_logits = model_output[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        
        # Should reduce sequence length by 1
        assert shift_logits.shape == (batch_size, seq_len - 1, vocab_size)
        assert shift_targets.shape == (batch_size, seq_len - 1)
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        loss = torch.tensor(2.0)
        perplexity = compute_perplexity(loss)
        
        expected = math.exp(2.0)
        assert abs(perplexity.item() - expected) < 0.01
    
    def test_compute_perplexity_zero_loss(self):
        """Test perplexity with zero loss."""
        loss = torch.tensor(0.0)
        perplexity = compute_perplexity(loss)
        
        assert perplexity.item() == 1.0
    
    def test_compute_perplexity_high_loss(self):
        """Test perplexity with high loss."""
        loss = torch.tensor(5.0)
        perplexity = compute_perplexity(loss)
        
        # High loss should give high perplexity
        assert perplexity.item() > 100


class TestOptimizer:
    """Test suite for optimizer utilities."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )
        return AtlasLM(config)
    
    def test_create_optimizer_basic(self, model):
        """Test basic optimizer creation."""
        optimizer = create_optimizer(model, learning_rate=1e-3)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # decay and no_decay groups
    
    def test_create_optimizer_learning_rate(self, model):
        """Test that learning rate is set correctly."""
        lr = 3e-4
        optimizer = create_optimizer(model, learning_rate=lr)
        
        # Both param groups should have the same LR
        for group in optimizer.param_groups:
            assert group['lr'] == lr
    
    def test_create_optimizer_weight_decay_groups(self, model):
        """Test that weight decay is applied selectively."""
        optimizer = create_optimizer(model, weight_decay=0.01)
        
        # First group should have weight decay
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
        
        # Second group should have no weight decay
        assert optimizer.param_groups[1]['weight_decay'] == 0.0
    
    def test_create_optimizer_parameters_split(self, model):
        """Test that parameters are split into decay/no_decay groups."""
        optimizer = create_optimizer(model)
        
        # Both groups should have parameters
        assert len(optimizer.param_groups[0]['params']) > 0
        assert len(optimizer.param_groups[1]['params']) > 0
    
    def test_get_optimizer_state(self, model):
        """Test getting optimizer state."""
        optimizer = create_optimizer(model, learning_rate=1e-3)
        state = get_optimizer_state(optimizer)
        
        assert 'learning_rates' in state
        assert 'num_param_groups' in state
        assert 'param_counts' in state
        
        assert len(state['learning_rates']) == 2
        assert state['num_param_groups'] == 2
        assert all(lr == 1e-3 for lr in state['learning_rates'])
    
    def test_clip_gradients(self, model):
        """Test gradient clipping."""
        # Create some gradients
        optimizer = create_optimizer(model)
        
        # Forward pass with dummy data
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        targets = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        
        loss = compute_lm_loss(logits, targets.reshape(-1, targets.shape[-1]))
        loss.backward()
        
        # Clip gradients
        grad_norm = clip_gradients(model, max_grad_norm=1.0)
        
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0
    
    def test_clip_gradients_effect(self, model):
        """Test that gradient clipping actually reduces gradient norm."""
        # Create large gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 10.0  # Large gradients
        
        # Get norm before clipping
        norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Reset gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 10.0
        
        # Clip to small value
        clip_gradients(model, max_grad_norm=1.0)
        
        # Check norm after clipping
        norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        assert norm_after <= 1.0 + 1e-5  # Allow small numerical error


class TestScheduler:
    """Test suite for learning rate scheduler."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a dummy optimizer."""
        model = torch.nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def test_create_scheduler_cosine(self, optimizer):
        """Test cosine scheduler creation."""
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type='cosine'
        )
        
        assert scheduler is not None
    
    def test_create_scheduler_linear(self, optimizer):
        """Test linear scheduler creation."""
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type='linear'
        )
        
        assert scheduler is not None
    
    def test_create_scheduler_constant(self, optimizer):
        """Test constant scheduler creation."""
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,
            scheduler_type='constant'
        )
        
        assert scheduler is not None
    
    def test_scheduler_warmup_phase(self, optimizer):
        """Test that learning rate increases during warmup."""
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100,
            scheduler_type='cosine'
        )
        
        lrs = []
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # LR should increase during warmup
        assert lrs[0] < lrs[50] < lrs[99]
        # Should reach initial LR at end of warmup
        assert abs(lrs[99] - initial_lr) < 1e-4
    
    def test_scheduler_cosine_decay(self, optimizer):
        """Test cosine decay schedule."""
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100,
            scheduler_type='cosine',
            min_lr_ratio=0.1
        )
        
        # Run through all steps
        for _ in range(1000):
            optimizer.step()
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # Final LR should be close to min_lr_ratio * initial_lr
        expected_min_lr = initial_lr * 0.1
        assert abs(final_lr - expected_min_lr) < initial_lr * 0.1
    
    def test_scheduler_linear_decay(self, optimizer):
        """Test linear decay schedule."""
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100,
            scheduler_type='linear',
            min_lr_ratio=0.1
        )
        
        # Run through all steps
        for _ in range(1000):
            optimizer.step()
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # Final LR should be close to min_lr_ratio * initial_lr
        expected_min_lr = initial_lr * 0.1
        assert abs(final_lr - expected_min_lr) < initial_lr * 0.1
    
    def test_scheduler_constant_after_warmup(self, optimizer):
        """Test constant LR after warmup."""
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100,
            scheduler_type='constant'
        )
        
        # Skip warmup
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        
        # LR should be constant after warmup
        lr_at_100 = optimizer.param_groups[0]['lr']
        
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        
        lr_at_200 = optimizer.param_groups[0]['lr']
        
        assert abs(lr_at_100 - initial_lr) < 1e-4
        assert abs(lr_at_200 - initial_lr) < 1e-4
    
    def test_scheduler_warmup_ratio(self, optimizer):
        """Test warmup ratio calculation."""
        initial_lr = optimizer.param_groups[0]['lr']
        
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=1000,
            warmup_ratio=0.1,  # Should be 100 steps
            scheduler_type='cosine'
        )
        
        # After 100 steps (warmup), should be at full LR
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert abs(lr_after_warmup - initial_lr) < 1e-4
    
    def test_scheduler_invalid_type(self, optimizer):
        """Test that invalid scheduler type raises error."""
        # Error should happen during creation now
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_scheduler(
                optimizer,
                num_training_steps=1000,
                scheduler_type='invalid'
            )


class TestTrainer:
    """Test suite for Trainer class."""
    
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=32,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
        )
        return AtlasLM(config)
    
    @pytest.fixture
    def tiny_dataloader(self):
        """Create a tiny dataloader for testing."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        num_samples = 8
        seq_len = 16
        vocab_size = 100
        
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        dataset = TensorDataset(input_ids)
        
        # Custom collate function
        def collate_fn(batch):
            return {'input_ids': torch.stack([item[0] for item in batch])}
        
        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    def test_trainer_initialization(self, tiny_model):
        """Test Trainer initialization."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        assert trainer.global_step == 0
        assert trainer.gradient_accumulation_steps == 1
        assert trainer.max_grad_norm == 1.0
        assert trainer.device == 'cpu'
    
    def test_trainer_single_step(self, tiny_model, tiny_dataloader):
        """Test single training step."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        # Get a batch
        batch = next(iter(tiny_dataloader))
        
        # Execute training step
        loss = trainer.train_step(batch, accumulation_step=0)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert trainer.global_step == 1
    
    def test_trainer_gradient_accumulation(self, tiny_model, tiny_dataloader):
        """Test gradient accumulation."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        accumulation_steps = 4
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            gradient_accumulation_steps=accumulation_steps,
            device='cpu',
        )
        
        batch = next(iter(tiny_dataloader))
        
        # First 3 accumulation steps should not update global_step
        for step in range(3):
            trainer.train_step(batch, accumulation_step=step)
            assert trainer.global_step == 0
        
        # 4th step should trigger update
        trainer.train_step(batch, accumulation_step=3)
        assert trainer.global_step == 1
    
    def test_trainer_with_scheduler(self, tiny_model, tiny_dataloader):
        """Test trainer with learning rate scheduler."""
        from atlas.training import Trainer, create_optimizer, create_scheduler
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=10,
            num_warmup_steps=2,
            scheduler_type='linear',
        )
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
        )
        
        batch = next(iter(tiny_dataloader))
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Execute a few steps
        for step in range(3):
            trainer.train_step(batch, accumulation_step=0)
        
        # LR should have changed
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr != initial_lr
    
    def test_trainer_gradient_clipping(self, tiny_model, tiny_dataloader):
        """Test gradient clipping."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-3)
        max_grad_norm = 0.5
        
        # Create trainer without auto-clipping to test manually
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            max_grad_norm=None,  # Disable auto-clipping
            gradient_accumulation_steps=2,  # Use accumulation to prevent immediate zero_grad
            device='cpu',
        )
        
        batch = next(iter(tiny_dataloader))
        
        # Execute training step (accumulation_step=0, won't update)
        trainer.train_step(batch, accumulation_step=0)
        
        # Gradients should exist now
        assert any(p.grad is not None for p in tiny_model.parameters())
        
        # Compute total gradient norm before clipping
        total_norm_before = 0.0
        for p in tiny_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # Apply gradient clipping manually
        from atlas.training import clip_gradients
        clip_gradients(tiny_model, max_grad_norm)
        
        # Compute total gradient norm after clipping
        total_norm_after = 0.0
        for p in tiny_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        # If gradients were large, norm should be clipped
        if total_norm_before > max_grad_norm:
            assert abs(total_norm_after - max_grad_norm) < 0.1
    
    def test_trainer_epoch(self, tiny_model, tiny_dataloader):
        """Test full epoch training."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        # Train for one epoch
        stats = trainer.train_epoch(tiny_dataloader, log_interval=2)
        
        assert 'loss' in stats
        assert 'perplexity' in stats
        assert 'tokens' in stats
        assert 'steps' in stats
        assert stats['loss'] > 0
        assert stats['perplexity'] > 0
        assert stats['tokens'] > 0
        assert trainer.global_step > 0
    
    def test_trainer_max_steps(self, tiny_model, tiny_dataloader):
        """Test training with max_steps limit."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        max_steps = 2
        trainer.train_epoch(tiny_dataloader, max_steps=max_steps, log_interval=1)
        
        assert trainer.global_step == max_steps
    
    def test_training_metrics(self, tiny_model, tiny_dataloader):
        """Test TrainingMetrics computation."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-3)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        batch = next(iter(tiny_dataloader))
        loss = trainer.train_step(batch, accumulation_step=0)
        
        metrics = trainer.get_current_metrics(loss)
        
        assert metrics.step == 1
        assert metrics.loss == loss
        assert metrics.perplexity > 0
        assert metrics.learning_rate == 1e-3
        assert metrics.tokens_per_second >= 0
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'step' in metrics_dict
        assert 'loss' in metrics_dict
    
    def test_trainer_loss_decreases(self, tiny_model):
        """Test that loss decreases on tiny overfitting test."""
        from atlas.training import Trainer, create_optimizer
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create single batch to overfit on
        input_ids = torch.randint(0, 100, (2, 16))
        dataset = TensorDataset(input_ids)
        
        def collate_fn(batch):
            return {'input_ids': torch.stack([item[0] for item in batch])}
        
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-3)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        # Train for a few steps on same batch
        batch = next(iter(dataloader))
        losses = []
        for _ in range(10):
            loss = trainer.train_step(batch, accumulation_step=0)
            losses.append(loss)
        
        # Loss should decrease (overfitting on single batch)
        assert losses[-1] < losses[0], "Loss should decrease when overfitting"
    
    def test_trainer_reset_metrics(self, tiny_model):
        """Test metric reset."""
        from atlas.training import Trainer, create_optimizer
        
        optimizer = create_optimizer(tiny_model, learning_rate=1e-4)
        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device='cpu',
        )
        
        # Manually set some metrics
        trainer.tokens_processed = 1000
        
        # Reset
        trainer.reset_metrics()
        
        assert trainer.tokens_processed == 0

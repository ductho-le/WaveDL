"""
WaveDL - Deep Learning for Wave-based Inverse Problems
=======================================================
Target Environment: NVIDIA HPC GPUs (Multi-GPU DDP) | PyTorch 2.x | Python 3.11+

A modular training framework for wave-based inverse problems and regression:
  1. HPC-Grade DDP Training: BF16/FP16 mixed precision with torch.compile support
  2. Dynamic Model Selection: Use --model flag to select any registered architecture
  3. Zero-Copy Data Engine: Memmap-backed datasets for large-scale training
  4. Physics-Aware Metrics: Real-time physical MAE with proper unscaling
  5. Robust Checkpointing: Resume training, periodic saves, and training curves
  6. Deep Observability: WandB integration with scatter analysis

Usage:
    # Recommended: Using the HPC helper script
    ./run_training.sh --model cnn --batch_size 128 --wandb
    
    # Or with direct accelerate launch
    accelerate launch train.py --model cnn --batch_size 128 --wandb

    # Multi-GPU with explicit config
    accelerate launch --num_processes=4 --mixed_precision=bf16 \
        train.py --model cnn --wandb --project_name "MyProject"

    # Resume from checkpoint
    accelerate launch train.py --model cnn --resume best_checkpoint --wandb

    # List available models
    python train.py --list_models

Note:
    For HPC clusters (Compute Canada, etc.), use run_training.sh which handles
    environment configuration automatically. Mixed precision is controlled via
    --precision flag (default: bf16).

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

# ==============================================================================
# ENVIRONMENT CONFIGURATION FOR HPC SYSTEMS
# ==============================================================================
# IMPORTANT: These must be set BEFORE matplotlib is imported to be effective
import os
os.environ.setdefault('MPLCONFIGDIR', os.getenv('TMPDIR', '/tmp') + '/matplotlib')
os.environ.setdefault('FONTCONFIG_PATH', '/etc/fonts')

import sys
import argparse
import logging
import pickle
import shutil
import time
import warnings
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator
from accelerate.utils import set_seed

# Local imports
from models import get_model, list_models, build_model
from utils import (
    MetricTracker,
    calc_pearson,
    plot_scientific_scatter,
    create_training_curves,
    get_lr,
    prepare_data,
    broadcast_early_stop,
    # New factory functions
    get_loss,
    list_losses,
    get_optimizer,
    list_optimizers,
    get_scheduler,
    list_schedulers,
    is_epoch_based,
    FIGURE_DPI,
)

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Filter non-critical warnings for cleaner training logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Universal DDP Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Selection
    parser.add_argument(
        '--model', type=str, default='cnn',
        help=f"Model architecture to train. Available: {list_models()}"
    )
    parser.add_argument(
        '--list_models', action='store_true',
        help="List all available models and exit"
    )
    
    # Configuration File
    parser.add_argument(
        '--config', type=str, default=None,
        help="Path to YAML config file. CLI args override config values."
    )
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="Maximum training epochs")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping norm")
    
    # Loss Function
    parser.add_argument(
        '--loss', type=str, default='mse',
        choices=['mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'weighted_mse'],
        help=f"Loss function for training. Available: {list_losses()}"
    )
    parser.add_argument('--huber_delta', type=float, default=1.0, help="Delta for Huber loss")
    parser.add_argument(
        '--loss_weights', type=str, default=None,
        help="Comma-separated weights for weighted_mse (e.g., '1.0,2.0,1.0')"
    )
    
    # Optimizer
    parser.add_argument(
        '--optimizer', type=str, default='adamw',
        choices=['adamw', 'adam', 'sgd', 'nadam', 'radam', 'rmsprop'],
        help=f"Optimizer for training. Available: {list_optimizers()}"
    )
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD/RMSprop")
    parser.add_argument('--nesterov', action='store_true', help="Use Nesterov momentum (SGD)")
    parser.add_argument('--betas', type=str, default='0.9,0.999', help="Betas for Adam variants (comma-separated)")
    
    # Learning Rate Scheduler
    parser.add_argument(
        '--scheduler', type=str, default='plateau',
        choices=['plateau', 'cosine', 'cosine_restarts', 'onecycle', 'step', 'multistep', 'exponential', 'linear_warmup'],
        help=f"LR scheduler. Available: {list_schedulers()}"
    )
    parser.add_argument('--scheduler_patience', type=int, default=10, help="Patience for ReduceLROnPlateau")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help="LR reduction factor")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Warmup epochs for linear_warmup")
    parser.add_argument('--step_size', type=int, default=30, help="Step size for StepLR")
    parser.add_argument(
        '--milestones', type=str, default=None,
        help="Comma-separated epochs for MultiStepLR (e.g., '30,60,90')"
    )
    
    # Data
    parser.add_argument('--data_path', type=str, default='train_data.npz', help="Path to NPZ dataset")
    parser.add_argument('--workers', type=int, default=0, help="DataLoader workers")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed")
    
    # Cross-Validation
    parser.add_argument('--cv', type=int, default=0, 
                        help="Enable K-fold cross-validation with K folds (0=disabled)")
    parser.add_argument('--cv_stratify', action='store_true',
                        help="Use stratified splitting for cross-validation")
    parser.add_argument('--cv_bins', type=int, default=10,
                        help="Number of bins for stratified CV (only with --cv_stratify)")
    
    # Checkpointing & Resume
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument('--save_every', type=int, default=50, help="Save checkpoint every N epochs (0=disable)")
    parser.add_argument('--output_dir', type=str, default='.', help="Output directory for checkpoints")
    parser.add_argument('--fresh', action='store_true', help="Force fresh training, ignore existing checkpoints")
    
    # Performance
    parser.add_argument('--compile', action='store_true', help="Enable torch.compile (PyTorch 2.x)")
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'],
                        help="Mixed precision mode")
    
    # Logging
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--project_name', type=str, default='DL-Training', help="WandB project name")
    parser.add_argument('--run_name', type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    return args, parser  # Returns (Namespace, ArgumentParser)


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def main():
    args, parser = parse_args()
    
    # Handle --list_models flag
    if args.list_models:
        print("Available models:")
        for name in list_models():
            ModelClass = get_model(name)
            # Get first non-empty docstring line
            if ModelClass.__doc__:
                lines = [l.strip() for l in ModelClass.__doc__.splitlines() if l.strip()]
                doc_first_line = lines[0] if lines else 'No description'
            else:
                doc_first_line = 'No description'
            print(f"  - {name}: {doc_first_line}")
        sys.exit(0)

    
    # Load and merge config file if provided
    if args.config:
        from utils.config import load_config, merge_config_with_args, validate_config
        print(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)
        
        # Validate config values
        warnings_list = validate_config(config)
        for w in warnings_list:
            print(f"  ⚠ {w}")
        
        # Merge config with CLI args (CLI takes precedence via parser defaults detection)
        args = merge_config_with_args(config, args, parser=parser)
    
    # Handle --cv flag (cross-validation mode)
    if args.cv > 0:
        print(f"🔄 Cross-Validation Mode: {args.cv} folds")
        from utils.cross_validation import run_cross_validation
        
        # Load data for CV
        data = np.load(args.data_path, allow_pickle=True)
        X = data['input_train']
        y = data['output_train']
        
        # Handle sparse matrices
        if hasattr(X[0], 'toarray'):
            X = np.stack([x.toarray() for x in X])
        
        # Normalize target shape: (N,) -> (N, 1) for consistency
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Determine input shape (spatial dimensions only, channel is added during training)
        # Data from NPZ: (N, *spatial) where spatial can be (L,), (H, W), or (D, H, W)
        # This matches prepare_data which adds channel dim: (N, 1, *spatial)
        in_shape = X.shape[1:]  # Always take all spatial dimensions
        
        # Run cross-validation
        run_cross_validation(
            X=X,
            y=y,
            model_name=args.model,
            in_shape=in_shape,
            out_size=y.shape[1],
            folds=args.cv,
            stratify=args.cv_stratify,
            stratify_bins=args.cv_bins,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            output_dir=args.output_dir,
            workers=args.workers,
            seed=args.seed,
        )
        return
    
    # ==========================================================================
    # 1. SYSTEM INITIALIZATION
    # ==========================================================================
    # Initialize Accelerator for DDP and mixed precision
    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="wandb" if args.wandb and WANDB_AVAILABLE else None
    )
    set_seed(args.seed)
    
    # Configure logging (rank 0 only prints to console)
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("Trainer")
    
    # Ensure output directory exists (critical for cache files, checkpoints, etc.)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        logger.info(f"🚀 Cluster Status: {accelerator.num_processes}x GPUs detected")
        logger.info(f"   Model: {args.model} | Precision: {args.precision} | Compile: {args.compile}")
        logger.info(f"   Loss: {args.loss} | Optimizer: {args.optimizer} | Scheduler: {args.scheduler}")
        logger.info(f"   Early Stopping Patience: {args.patience} epochs")
        if args.save_every > 0:
            logger.info(f"   Periodic Checkpointing: Every {args.save_every} epochs")
        if args.resume:
            logger.info(f"   📂 Resuming from: {args.resume}")
        
        # Initialize WandB
        if args.wandb and WANDB_AVAILABLE:
            accelerator.init_trackers(
                project_name=args.project_name,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.run_name or f"{args.model}_run"}}
            )
    
    # ==========================================================================
    # 2. DATA & MODEL LOADING
    # ==========================================================================
    train_dl, val_dl, scaler, in_shape, out_dim = prepare_data(
        args, logger, accelerator, cache_dir=args.output_dir
    )
    
    # Build model using registry
    model = build_model(args.model, in_shape=in_shape, out_size=out_dim)
    
    if accelerator.is_main_process:
        param_info = model.parameter_summary()
        logger.info(f"   Model Parameters: {param_info['trainable_parameters']:,} trainable")
        logger.info(f"   Model Size: {param_info['total_mb']:.2f} MB")
    
    # Optional WandB model watching
    if args.wandb and WANDB_AVAILABLE and accelerator.is_main_process:
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Torch 2.0 compilation (requires compatible Triton on GPU)
    if args.compile:
        compile_enabled = False
        try:
            # Test if Triton is available AND compatible with this PyTorch version
            # PyTorch needs triton_key from triton.compiler.compiler
            from triton.compiler.compiler import triton_key
            model = torch.compile(model)
            compile_enabled = True
            if accelerator.is_main_process:
                logger.info("   ✔ torch.compile enabled (Triton backend)")
        except ImportError as e:
            if accelerator.is_main_process:
                if "triton" in str(e).lower():
                    logger.warning(
                        "   ⚠ Triton not installed or incompatible version - torch.compile disabled. "
                        "Training will proceed without compilation."
                    )
                else:
                    logger.warning(f"   ⚠ torch.compile setup failed: {e}. Continuing without compilation.")
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(f"   ⚠ torch.compile failed: {e}. Continuing without compilation.")
    
    # ==========================================================================
    # 2.5. OPTIMIZER, SCHEDULER & LOSS CONFIGURATION
    # ==========================================================================
    # Parse comma-separated arguments with validation
    try:
        betas_list = [float(x.strip()) for x in args.betas.split(',')]
        if len(betas_list) != 2:
            raise ValueError(f"--betas must have exactly 2 values, got {len(betas_list)}")
        if not all(0.0 <= b < 1.0 for b in betas_list):
            raise ValueError(f"--betas values must be in [0, 1), got {betas_list}")
        betas = tuple(betas_list)
    except ValueError as e:
        raise ValueError(f"Invalid --betas format '{args.betas}': {e}. Expected format: '0.9,0.999'")
    
    loss_weights = None
    if args.loss_weights:
        loss_weights = [float(x.strip()) for x in args.loss_weights.split(',')]
    milestones = None
    if args.milestones:
        milestones = [int(x.strip()) for x in args.milestones.split(',')]
    
    # Create optimizer using factory
    optimizer = get_optimizer(
        name=args.optimizer,
        params=model.get_optimizer_groups(args.lr, args.weight_decay),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
        betas=betas,
    )
    
    # Create loss function using factory
    criterion = get_loss(
        name=args.loss,
        weights=loss_weights,
        delta=args.huber_delta,
    )
    # Move criterion to device (important for WeightedMSELoss buffer)
    criterion = criterion.to(accelerator.device)
    
    # Track if scheduler should step per batch (OneCycleLR) or per epoch
    scheduler_step_per_batch = not is_epoch_based(args.scheduler)
    
    # ==========================================================================
    # DDP Preparation Strategy:
    # - For batch-based schedulers (OneCycleLR): prepare DataLoaders first to get
    #   the correct sharded batch count, then create scheduler
    # - For epoch-based schedulers: create scheduler before prepare (no issue)
    # ==========================================================================
    if scheduler_step_per_batch:
        # BATCH-BASED SCHEDULER (e.g., OneCycleLR)
        # Prepare model, optimizer, dataloaders FIRST to get sharded loader length
        model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl
        )
        
        # Now create scheduler with the CORRECT sharded steps_per_epoch
        steps_per_epoch = len(train_dl)  # Post-DDP sharded length
        scheduler = get_scheduler(
            name=args.scheduler,
            optimizer=optimizer,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr=args.min_lr,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            gamma=args.scheduler_factor,  # For Step/MultiStep/Exponential schedulers
            step_size=args.step_size,
            milestones=milestones,
            warmup_epochs=args.warmup_epochs,
        )
        # Prepare scheduler separately (Accelerator wraps it for state saving)
        scheduler = accelerator.prepare(scheduler)
    else:
        # EPOCH-BASED SCHEDULER (plateau, cosine, step, etc.)
        # No batch count dependency - create scheduler before prepare
        scheduler = get_scheduler(
            name=args.scheduler,
            optimizer=optimizer,
            epochs=args.epochs,
            steps_per_epoch=None,
            min_lr=args.min_lr,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            gamma=args.scheduler_factor,  # For Step/MultiStep/Exponential schedulers
            step_size=args.step_size,
            milestones=milestones,
            warmup_epochs=args.warmup_epochs,
        )
        # Prepare everything together
        model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
            model, optimizer, train_dl, val_dl, scheduler
        )
    
    # ==========================================================================
    # 3. AUTO-RESUME / RESUME FROM CHECKPOINT
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    patience_ctr = 0
    history: List[Dict[str, Any]] = []
    
    # Define checkpoint paths
    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint")
    complete_flag_path = os.path.join(args.output_dir, "training_complete.flag")
    
    # Auto-resume logic (if not --fresh and no explicit --resume)
    if not args.fresh and args.resume is None:
        if os.path.exists(complete_flag_path):
            # Training already completed
            if accelerator.is_main_process:
                logger.info("✅ Training already completed (early stopping). Use --fresh to retrain.")
            return  # Exit gracefully
        elif os.path.exists(best_ckpt_path):
            # Incomplete training found - auto-resume
            args.resume = best_ckpt_path
            if accelerator.is_main_process:
                logger.info(f"🔄 Auto-resuming from: {best_ckpt_path}")
    
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"🔄 Loading checkpoint from: {args.resume}")
            accelerator.load_state(args.resume)
            
            # Restore training metadata
            meta_path = os.path.join(args.resume, "training_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                start_epoch = meta.get("epoch", 0)
                best_val_loss = meta.get("best_val_loss", float('inf'))
                patience_ctr = meta.get("patience_ctr", 0)
                logger.info(f"   ✅ Restored: Epoch {start_epoch}, Best Loss: {best_val_loss:.6f}")
            else:
                logger.warning("   ⚠️ training_meta.pkl not found, starting from epoch 0")
            
            # Restore history
            history_path = os.path.join(args.output_dir, "training_history.csv")
            if os.path.exists(history_path):
                history = pd.read_csv(history_path).to_dict('records')
                logger.info(f"   ✅ Loaded {len(history)} epochs from history")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    
    # ==========================================================================
    # 4. PHYSICAL METRIC SETUP
    # ==========================================================================
    # Physical MAE = normalized MAE * scaler.scale_
    phys_scale = torch.tensor(scaler.scale_, device=accelerator.device, dtype=torch.float32)
    
    # ==========================================================================
    # 5. TRAINING LOOP
    # ==========================================================================
    # Dynamic console header
    if accelerator.is_main_process:
        base_cols = ["Ep", "TrnLoss", "ValLoss", "R2", "PCC", "GradN", "LR", "MAE_Avg"]
        param_cols = [f"MAE_P{i}" for i in range(out_dim)]
        header = "{:<4} | {:<8} | {:<8} | {:<6} | {:<6} | {:<6} | {:<8} | {:<8}".format(*base_cols)
        header += " | " + " | ".join([f"{c:<8}" for c in param_cols])
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("=" * len(header))
    
    try:
        training_start_time = time.time()
        total_training_time = 0.0
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # ==================== TRAINING PHASE ====================
            model.train()
            train_loss_sum = 0.0
            train_samples = 0
            grad_norm_tracker = MetricTracker()
            
            pbar = tqdm(
                train_dl,
                disable=not accelerator.is_main_process,
                leave=False,
                desc=f"Epoch {epoch+1}"
            )
            
            for x, y in pbar:
                with accelerator.accumulate(model):
                    pred = model(x)
                    loss = criterion(pred, y)
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                        if grad_norm is not None:
                            grad_norm_tracker.update(grad_norm.item())
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Per-batch LR scheduling (e.g., OneCycleLR)
                    if scheduler_step_per_batch:
                        scheduler.step()
                    
                    train_loss_sum += loss.item() * x.size(0)
                    train_samples += x.size(0)
                    
                    if accelerator.is_main_process:
                        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # Synchronize training metrics across GPUs
            global_loss = accelerator.reduce(
                torch.tensor([train_loss_sum], device=accelerator.device), reduction="sum"
            ).item()
            global_samples = accelerator.reduce(
                torch.tensor([train_samples], device=accelerator.device), reduction="sum"
            ).item()
            avg_train_loss = global_loss / global_samples
            
            # ==================== VALIDATION PHASE ====================
            model.eval()
            val_loss_sum = 0.0
            val_mae_sum = torch.zeros(out_dim, device=accelerator.device)
            val_samples = 0
            
            preds_buffer = []
            targets_buffer = []
            
            with torch.inference_mode():
                for x, y in val_dl:
                    pred = model(x)
                    loss = criterion(pred, y)
                    
                    val_loss_sum += loss.item() * x.size(0)
                    val_samples += x.size(0)
                    
                    # Physical MAE
                    mae_batch = torch.abs((pred - y) * phys_scale).sum(dim=0)
                    val_mae_sum += mae_batch
                    
                    # Gather for metrics
                    all_pred = accelerator.gather_for_metrics(pred)
                    all_y = accelerator.gather_for_metrics(y)
                    
                    if accelerator.is_main_process:
                        preds_buffer.append(all_pred.cpu())
                        targets_buffer.append(all_y.cpu())
            
            # Synchronize validation metrics
            val_metrics = torch.cat([
                torch.tensor([val_loss_sum], device=accelerator.device),
                val_mae_sum
            ])
            val_metrics_sync = accelerator.reduce(val_metrics, reduction="sum")
            
            total_val_samples = accelerator.reduce(
                torch.tensor([val_samples], device=accelerator.device), reduction="sum"
            ).item()
            
            avg_val_loss = val_metrics_sync[0].item() / total_val_samples
            # Cast to float32 before numpy (bf16 tensors can't convert directly)
            avg_mae_per_param = (val_metrics_sync[1:] / total_val_samples).float().cpu().numpy()
            avg_mae = avg_mae_per_param.mean()
            
            # ==================== LOGGING & CHECKPOINTING ====================
            if accelerator.is_main_process:
                # Scientific metrics - cast to float32 before numpy (bf16 can't convert)
                y_pred = torch.cat(preds_buffer).float().numpy()
                y_true = torch.cat(targets_buffer).float().numpy()
                
                # Trim DDP padding
                real_len = len(val_dl.dataset)
                if len(y_pred) > real_len:
                    y_pred = y_pred[:real_len]
                    y_true = y_true[:real_len]
                
                from sklearn.metrics import r2_score
                # Guard against tiny validation sets (R² undefined for <2 samples)
                if len(y_true) >= 2:
                    r2 = r2_score(y_true, y_pred)
                else:
                    r2 = float('nan')
                pcc = calc_pearson(y_true, y_pred)
                current_lr = get_lr(optimizer)
                
                # Update history
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                total_training_time += epoch_time
                
                epoch_stats = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_r2": r2,
                    "val_pearson": pcc,
                    "val_mae_avg": avg_mae,
                    "grad_norm": grad_norm_tracker.avg,
                    "lr": current_lr,
                    "epoch_time": round(epoch_time, 2),
                    "total_time": round(total_training_time, 2)
                }
                for i, mae in enumerate(avg_mae_per_param):
                    epoch_stats[f"MAE_Phys_P{i}"] = mae
                
                history.append(epoch_stats)
                pd.DataFrame(history).to_csv(
                    os.path.join(args.output_dir, "training_history.csv"), index=False
                )
                
                # Console display
                base_str = "{:<4} | {:<8.4f} | {:<8.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<8.2e} | {:<8.4f}".format(
                    epoch+1, avg_train_loss, avg_val_loss, r2, pcc, grad_norm_tracker.avg, current_lr, avg_mae
                )
                param_str = " | ".join([f"{m:<8.4f}" for m in avg_mae_per_param])
                logger.info(f"{base_str} | {param_str}")
                
                # WandB logging
                if args.wandb and WANDB_AVAILABLE:
                    log_dict = {
                        "main/train_loss": avg_train_loss,
                        "main/val_loss": avg_val_loss,
                        "metrics/r2_score": r2,
                        "metrics/pearson_corr": pcc,
                        "metrics/mae_avg": avg_mae,
                        "system/grad_norm": grad_norm_tracker.avg,
                        "hyper/lr": current_lr
                    }
                    for i, mae in enumerate(avg_mae_per_param):
                        log_dict[f"mae_detailed/P{i}"] = mae
                    
                    # Periodic scatter plots
                    if (epoch % 5 == 0) or (avg_val_loss < best_val_loss):
                        real_true = scaler.inverse_transform(y_true)
                        real_pred = scaler.inverse_transform(y_pred)
                        fig = plot_scientific_scatter(real_true, real_pred)
                        log_dict["plots/scatter_analysis"] = wandb.Image(fig)
                        plt.close(fig)
                    
                    accelerator.log(log_dict)
            
            # ==========================================================================
            # DDP-SAFE CHECKPOINT LOGIC
            # ==========================================================================
            # Step 1: Determine if this is the best epoch (BEFORE updating best_val_loss)
            is_best_epoch = False
            if accelerator.is_main_process:
                if avg_val_loss < best_val_loss:
                    is_best_epoch = True
            
            # Step 2: Broadcast decision to all ranks (required for save_state)
            is_best_epoch = broadcast_early_stop(is_best_epoch, accelerator)
            
            # Step 3: Save checkpoint with all ranks participating
            if is_best_epoch:
                ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
                accelerator.save_state(ckpt_dir)  # All ranks must call this
                
                # Step 4: Rank 0 handles metadata and updates tracking variables
                if accelerator.is_main_process:
                    best_val_loss = avg_val_loss  # Update AFTER checkpoint saved
                    patience_ctr = 0
                    
                    with open(os.path.join(ckpt_dir, "training_meta.pkl"), "wb") as f:
                        pickle.dump({
                            "epoch": epoch + 1,
                            "best_val_loss": best_val_loss,
                            "patience_ctr": patience_ctr,
                            # Model info for auto-detection during inference
                            "model_name": args.model,
                            "in_shape": in_shape,
                            "out_dim": out_dim,
                        }, f)
                    
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(
                        unwrapped.state_dict(),
                        os.path.join(args.output_dir, "best_model_weights.pth")
                    )
                    
                    # Copy scaler to checkpoint for portability
                    scaler_src = os.path.join(args.output_dir, "scaler.pkl")
                    scaler_dst = os.path.join(ckpt_dir, "scaler.pkl")
                    if os.path.exists(scaler_src) and not os.path.exists(scaler_dst):
                        shutil.copy2(scaler_src, scaler_dst)
                    
                    logger.info(f"   💾 Best model saved (val_loss: {best_val_loss:.6f})")
            else:
                if accelerator.is_main_process:
                    patience_ctr += 1
                
            # Periodic checkpoint (all ranks participate in save_state)
            periodic_checkpoint_needed = args.save_every > 0 and (epoch + 1) % args.save_every == 0
            if periodic_checkpoint_needed:
                ckpt_name = f"epoch_{epoch+1}_checkpoint"
                ckpt_dir = os.path.join(args.output_dir, ckpt_name)
                accelerator.save_state(ckpt_dir)  # All ranks participate
                
                if accelerator.is_main_process:
                    with open(os.path.join(ckpt_dir, "training_meta.pkl"), "wb") as f:
                        pickle.dump({
                            "epoch": epoch + 1,
                            "best_val_loss": best_val_loss,
                            "patience_ctr": patience_ctr,
                            # Model info for auto-detection during inference
                            "model_name": args.model,
                            "in_shape": in_shape,
                            "out_dim": out_dim,
                        }, f)
                    logger.info(f"   📁 Periodic checkpoint: {ckpt_name}")
            
            # Learning rate scheduling (epoch-based schedulers only)
            if not scheduler_step_per_batch:
                if args.scheduler == 'plateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # DDP-safe early stopping
            should_stop = patience_ctr >= args.patience if accelerator.is_main_process else False
            if broadcast_early_stop(should_stop, accelerator):
                if accelerator.is_main_process:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1} (patience={args.patience})")
                    # Create completion flag to prevent auto-resume
                    with open(os.path.join(args.output_dir, "training_complete.flag"), "w") as f:
                        f.write(f"Training completed via early stopping at epoch {epoch+1}\n")
                break
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving emergency checkpoint...")
        accelerator.save_state(os.path.join(args.output_dir, "interrupted_checkpoint"))
    
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        raise
    
    else:
        # Training completed normally (reached max epochs without early stopping)
        # Create completion flag to prevent auto-resume on re-run
        if accelerator.is_main_process:
            if not os.path.exists(complete_flag_path):
                with open(complete_flag_path, "w") as f:
                    f.write(f"Training completed normally after {args.epochs} epochs\n")
                logger.info(f"✅ Training completed after {args.epochs} epochs")
    
    finally:
        # Generate training curves plot (PNG + SVG)
        if accelerator.is_main_process and len(history) > 0:
            try:
                fig = create_training_curves(history, show_lr=True)
                for fmt in ['png', 'svg']:
                    fig.savefig(
                        os.path.join(args.output_dir, f"training_curves.{fmt}"),
                        dpi=FIGURE_DPI, bbox_inches='tight'
                    )
                plt.close(fig)
                logger.info(f"✔ Saved: training_curves.png, training_curves.svg")
            except Exception as e:
                logger.warning(f"Could not generate training curves: {e}")
        
        if args.wandb and WANDB_AVAILABLE:
            accelerator.end_training()
        
        # Clean up distributed process group to prevent resource leak warning
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        logger.info("Training completed.")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()

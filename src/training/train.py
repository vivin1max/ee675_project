import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import wandb

# Set matplotlib to non-interactive backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import models
from ..models.baseline_gru import BaselineGRU
from ..models.lstm_ctc import LSTMCTC
from ..models.transformer_ctc import TransformerCTC

# Import data utilities
from ..dataio.dataset import load_dataset, collate_fn

# Import training utilities
from .utils import (
    WhiteNoiseAugmentation,
    ConstantOffsetAugmentation,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_device,
)
from .scheduler_utils import get_scheduler, get_current_lr
from .fastemit_loss import FastEmitCTCLoss, CTCLossWithOptions
from .coordinated_dropout import apply_coordinated_dropout  # Ablation experiment

# Import evaluation utilities
from ..evaluation.metrics import compute_cer, compute_wer
from ..dataio.preprocessing import PHONE_DEF_SIL


def create_model(config: Dict, device: str) -> nn.Module:

    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'baseline_gru':
        model = BaselineGRU(**model_params, device=device)
    elif model_type == 'lstm':
        model = LSTMCTC(**model_params, device=device)
    elif model_type == 'transformer':
        model = TransformerCTC(**model_params, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    config: Dict,
    device: str,
    writer: SummaryWriter,
    global_step: int,
) -> Tuple[float, int]:

    model.train()
    
    # Data augmentation
    white_noise = WhiteNoiseAugmentation(config['training'].get('white_noise_sd', 0.8))
    constant_offset = ConstantOffsetAugmentation(config['training'].get('constant_offset_sd', 0.2))
    
    # Coordinated dropout (ablation experiment - NOT in baseline)
    coord_dropout_p = config['training'].get('coordinated_dropout', 0.0)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for X, y, X_len, y_len, dayIdx in pbar:
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )
        
        # Apply data augmentation
        X = white_noise(X, device)
        X = constant_offset(X, device)
        
        # Apply coordinated dropout if enabled (ablation only)
        if coord_dropout_p > 0:
            X = apply_coordinated_dropout(X, p=coord_dropout_p, training=True)
        
        # Forward pass
        pred = model(X, dayIdx)
        
        # Compute output lengths after stride/kernel
        output_lengths = model.compute_output_length(X_len)
        
        # Compute CTC loss
        # CTC expects (T, N, C) format
        loss = loss_fn(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            output_lengths,
            y_len,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if specified
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']
            )
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{get_current_lr(optimizer):.6f}"
        })
        
        # TensorBoard logging
        if global_step % config['logging'].get('log_interval', 10) == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', get_current_lr(optimizer), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def phoneme_ids_to_text(phoneme_ids: np.ndarray) -> str:

    phonemes = []
    for pid in phoneme_ids:
        if 1 <= pid <= len(PHONE_DEF_SIL):
            phonemes.append(PHONE_DEF_SIL[pid - 1])
    return ' '.join(phonemes)


def phoneme_text_to_words(phoneme_text: str) -> list:

    # Split on SIL to get word boundaries
    words = phoneme_text.split(' SIL ')
    # Remove empty strings and strip whitespace
    words = [w.strip() for w in words if w.strip()]
    return words


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    config: Dict,
    device: str,
) -> Tuple[float, float, float]:

    model.eval()
    
    total_loss = 0.0
    total_edit_distance = 0
    total_seq_length = 0
    total_word_edit_distance = 0
    total_word_count = 0
    num_batches = 0
    
    with torch.no_grad():
        for X, y, X_len, y_len, dayIdx in tqdm(test_loader, desc="Evaluating", leave=False):
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                dayIdx.to(device),
            )
            
            # Forward pass
            pred = model(X, dayIdx)
            output_lengths = model.compute_output_length(X_len)
            
            # Compute loss
            loss = loss_fn(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                output_lengths,
                y_len,
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute CER and WER using greedy decoding
            for i in range(pred.shape[0]):
                pred_seq = torch.argmax(pred[i, :output_lengths[i], :], dim=-1)
                pred_seq = torch.unique_consecutive(pred_seq)
                pred_seq = pred_seq[pred_seq != 0]  # Remove blanks
                
                true_seq = y[i, :y_len[i]]
                
                # Compute CER (character-level edit distance)
                edit_dist, seq_len = compute_cer(
                    true_seq.cpu().numpy(),
                    pred_seq.cpu().numpy()
                )
                total_edit_distance += edit_dist
                total_seq_length += seq_len
                
                # Compute WER (word-level edit distance)
                try:
                    # Convert phoneme IDs to text
                    true_text = phoneme_ids_to_text(true_seq.cpu().numpy())
                    pred_text = phoneme_ids_to_text(pred_seq.cpu().numpy())
                    
                    # Split into words using SIL markers
                    true_words = phoneme_text_to_words(true_text)
                    pred_words = phoneme_text_to_words(pred_text)
                    
                    # Compute word-level edit distance
                    if len(true_words) > 0:
                        word_edit_dist, word_count = compute_wer(true_words, pred_words)
                        total_word_edit_distance += word_edit_dist
                        total_word_count += word_count
                except Exception as e:
                    # If WER computation fails, skip this sample
                    pass
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    cer = total_edit_distance / total_seq_length if total_seq_length > 0 else 1.0
    wer = total_word_edit_distance / total_word_count if total_word_count > 0 else 1.0
    
    return avg_loss, cer, wer


def plot_training_curves(history: dict, output_dir: Path, use_wandb: bool = False):

    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if len(history['test_loss']) > 0:
        test_epochs = [i for i in epochs if i % (len(epochs) // len(history['test_loss'])) == 0][:len(history['test_loss'])]
        ax1.plot(test_epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # CER curve
    if len(history['cer']) > 0:
        cer_epochs = [i for i in epochs if i % (len(epochs) // len(history['cer'])) == 0][:len(history['cer'])]
        ax2.plot(cer_epochs, history['cer'], 'g-', label='Test CER', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('CER', fontsize=12)
        ax2.set_title('Character Error Rate', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({"training_curves": wandb.Image(str(save_path))})
    
    plt.close()


def plot_gradient_norms(model: nn.Module, output_dir: Path, epoch: int, use_wandb: bool = False):

    grad_norms = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            layer_names.append(name)
    
    if len(grad_norms) == 0:
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(grad_norms)), grad_norms, color='steelblue')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'Gradient Norms (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / f'gradient_norms_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "gradients/norms": wandb.Image(str(save_path)),
            "gradients/max_norm": max(grad_norms),
            "gradients/mean_norm": np.mean(grad_norms),
        })
    
    plt.close()


def plot_learning_rate_schedule(history: dict, output_dir: Path, use_wandb: bool = False):

    if 'learning_rate' not in history or len(history['learning_rate']) == 0:
        return
    
    epochs = range(1, len(history['learning_rate']) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['learning_rate'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / 'learning_rate_schedule.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning rate schedule to {save_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({"learning_rate/schedule": wandb.Image(str(save_path))})
    
    plt.close()


def plot_loss_comparison(history: dict, output_dir: Path, use_wandb: bool = False):

    if len(history['train_loss']) == 0:
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with train and test loss
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train loss
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', 
            linewidth=2, markersize=4, alpha=0.8)
    
    # Plot test loss if available
    if len(history['test_loss']) > 0:
        test_epochs = [i for i in epochs if i % (len(epochs) // len(history['test_loss'])) == 0][:len(history['test_loss'])]
        ax.plot(test_epochs, history['test_loss'], 'r-s', label='Test Loss', 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('CTC Loss', fontsize=13)
    ax.set_title('Training Progress: Loss Over Time', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add min/max annotations
    if len(history['train_loss']) > 0:
        min_train = min(history['train_loss'])
        min_epoch = history['train_loss'].index(min_train) + 1
        ax.annotate(f'Min Train: {min_train:.3f}', 
                   xy=(min_epoch, min_train), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='blue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / 'loss_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss comparison to {save_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({"plots/loss_comparison": wandb.Image(str(save_path))})
    
    plt.close()


def plot_cer_progress(history: dict, output_dir: Path, use_wandb: bool = False):

    if 'cer' not in history or len(history['cer']) == 0:
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    cer_epochs = [i for i in epochs if i % (len(epochs) // len(history['cer'])) == 0][:len(history['cer'])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot CER
    ax.plot(cer_epochs, history['cer'], 'g-o', label='Character Error Rate', 
            linewidth=2.5, markersize=6, alpha=0.8)
    
    # Fill area under curve
    ax.fill_between(cer_epochs, history['cer'], alpha=0.2, color='green')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('CER (lower is better)', fontsize=13)
    ax.set_title('Model Performance: Character Error Rate', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add best CER annotation
    if len(history['cer']) > 0:
        best_cer = min(history['cer'])
        best_epoch_idx = history['cer'].index(best_cer)
        best_epoch = cer_epochs[best_epoch_idx]
        ax.annotate(f'Best CER: {best_cer:.4f}', 
                   xy=(best_epoch, best_cer), 
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / 'cer_progress.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved CER progress to {save_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({"plots/cer_progress": wandb.Image(str(save_path))})
    
    plt.close()


def plot_wer_progress(history: dict, output_dir: Path, use_wandb: bool = False):

    if 'wer' not in history or len(history['wer']) == 0:
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    wer_epochs = [i for i in epochs if i % (len(epochs) // len(history['wer'])) == 0][:len(history['wer'])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot WER
    ax.plot(wer_epochs, history['wer'], 'b-o', label='Word Error Rate', 
            linewidth=2.5, markersize=6, alpha=0.8)
    
    # Fill area under curve
    ax.fill_between(wer_epochs, history['wer'], alpha=0.2, color='blue')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('WER (lower is better)', fontsize=13)
    ax.set_title('Model Performance: Word Error Rate', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add best WER annotation
    if len(history['wer']) > 0:
        best_wer = min(history['wer'])
        best_epoch_idx = history['wer'].index(best_wer)
        best_epoch = wer_epochs[best_epoch_idx]
        ax.annotate(f'Best WER: {best_wer:.4f}', 
                   xy=(best_epoch, best_wer), 
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, color='darkblue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
    
    plt.tight_layout()
    
    # Save to disk
    save_path = output_dir / 'wer_progress.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved WER progress to {save_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({"plots/wer_progress": wandb.Image(str(save_path))})
    
    plt.close()


def train(config_path: str, resume_from: str = None, override_epochs: int = None):

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['training'].get('seed', 0))
    
    # Setup device
    device = get_device(prefer_cuda=config['training'].get('use_cuda', True))
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Weights & Biases
    use_wandb = config['training'].get('use_wandb', True)
    if use_wandb:
        try:
            wandb.init(
                project=config['training'].get('wandb_project', 'speech-bci'),
                name=config.get('experiment_name', 'unnamed'),
                config=config,
                dir=str(output_dir),
                tags=[config['model']['type'], config['training']['scheduler']['type']],
                notes=f"Training {config.get('experiment_name', 'unnamed')} with {config['model']['type']}",
            )
            print(f"Weights & Biases initialized: {wandb.run.url}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            use_wandb = False
    
    # Setup TensorBoard 
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Print experiment configuration
    print(f"\n{'='*60}")
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Model type: {config['model']['type']}")
    coord_dropout = config['training'].get('coordinated_dropout', 0.0)
    if coord_dropout > 0:
        print(f"Coordinated dropout: {coord_dropout} (ABLATION)")
    else:
        print("Coordinated dropout: disabled (baseline)")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, test_loader, loaded_data = load_dataset(
        config['data']['preprocessed_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', True),
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    print(f"Model type: {config['model']['type']}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Print model-specific info
    if hasattr(model, 'use_layernorm') and model.use_layernorm:
        print("  ✓ LayerNorm enabled (ABLATION)")
    
    print(model.get_model_summary())
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr_start'],
        betas=tuple(config['training'].get('betas', [0.9, 0.999])),
        eps=config['training'].get('eps', 0.1),
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )
    
    # Create learning rate scheduler
    scheduler = get_scheduler(
        config['training']['scheduler']['type'],
        optimizer,
        **config['training']['scheduler'].get('params', {})
    )
    print(f"Using LR scheduler: {config['training']['scheduler']['type']}")
    
    # Create loss function
    fastemit_lambda = config['training'].get('fastemit_lambda', 0.0)
    if fastemit_lambda > 0:
        loss_fn = FastEmitCTCLoss(
            blank=0,
            reduction='mean',
            zero_infinity=True,
            fastemit_lambda=fastemit_lambda,
        )
        print(f"Using FastEmit CTC loss with lambda={fastemit_lambda}")
    else:
        loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        print("Using standard CTC loss")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_cer = float('inf')
    
    # Initialize training history for plotting
    history = {
        'train_loss': [],
        'test_loss': [],
        'cer': [],
        'wer': [],
        'learning_rate': []
    }
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint_info = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = checkpoint_info['epoch']
        global_step = checkpoint_info['batch']
        best_cer = checkpoint_info['best_cer']
    
    # Training loop
    print("\nStarting training...")
    num_epochs = override_epochs if override_epochs is not None else config['training']['num_epochs']
    eval_interval = config['training'].get('eval_interval', 1)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, config, device, writer, global_step
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        
        # Store in history
        history['train_loss'].append(train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
            }, step=global_step)
        
        # Evaluate
        if (epoch + 1) % eval_interval == 0:
            test_loss, cer, wer = evaluate(model, test_loader, loss_fn, config, device)
            
            print(f"Test Loss: {test_loss:.4f}")
            print(f"CER: {cer:.4f}")
            print(f"WER: {wer:.4f}")
            
            writer.add_scalar('epoch/test_loss', test_loss, epoch)
            writer.add_scalar('epoch/cer', cer, epoch)
            writer.add_scalar('epoch/wer', wer, epoch)
            
            # Store in history
            history['test_loss'].append(test_loss)
            history['cer'].append(cer)
            history['wer'].append(wer)
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'test/loss': test_loss,
                    'test/cer': cer,
                    'test/wer': wer,
                }, step=global_step)
            
            # Generate and log plots periodically (every 5 epochs, at end, or if only 2 epochs)
            plot_interval = 5 if num_epochs > 10 else 1
            if (epoch + 1) % plot_interval == 0 or epoch == num_epochs - 1:
                # Plot loss comparison
                plot_loss_comparison(history, output_dir, use_wandb=use_wandb)
                
                # Plot CER progress
                plot_cer_progress(history, output_dir, use_wandb=use_wandb)
                
                # Plot WER progress
                plot_wer_progress(history, output_dir, use_wandb=use_wandb)
                
                # Plot learning rate schedule
                plot_learning_rate_schedule(history, output_dir, use_wandb=use_wandb)
            
            # Plot gradient norms (every 10 epochs or at the end)
            if use_wandb and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
                plot_gradient_norms(model, output_dir, epoch + 1, use_wandb=True)
            
            # Save best model
            if cer < best_cer:
                best_cer = cer
                save_path = output_dir / 'best_model.pt'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, best_cer, str(save_path)
                )
                print(f"✓ Saved best model (CER: {best_cer:.4f})")
                
                # Log best model to wandb
                if use_wandb:
                    wandb.run.summary['best_cer'] = best_cer
                    wandb.run.summary['best_epoch'] = epoch + 1
                    # Try to save model file (may fail on Windows without admin privileges)
                    try:
                        wandb.save(str(save_path), policy='now')
                    except (OSError, PermissionError) as e:
                        print(f"Note: Could not upload model to wandb (Windows symlink restriction): {e}")
                        print(f"Model saved locally at: {save_path}")
        
        # Save checkpoint periodically
        if (epoch + 1) % config['training'].get('save_interval', 10) == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, best_cer, str(save_path)
            )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_loss, cer, wer = evaluate(model, test_loader, loss_fn, config, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final CER: {cer:.4f}")
    print(f"Final WER: {wer:.4f}")
    print(f"Best CER: {best_cer:.4f}")
    
    # Generate all final plots
    if len(history['train_loss']) > 0:
        print("\nGenerating final plots...")
        plot_training_curves(history, output_dir, use_wandb=use_wandb)
        plot_loss_comparison(history, output_dir, use_wandb=use_wandb)
        plot_cer_progress(history, output_dir, use_wandb=use_wandb)
        plot_wer_progress(history, output_dir, use_wandb=use_wandb)
        plot_learning_rate_schedule(history, output_dir, use_wandb=use_wandb)
    
    # Log final results to wandb
    if use_wandb:
        wandb.run.summary['final_test_loss'] = test_loss
        wandb.run.summary['final_cer'] = cer
        wandb.run.summary['final_wer'] = wer
        
        # Create summary metrics table
        summary_data = []
        for i, (tl, te, c, w, lr) in enumerate(zip(
            history['train_loss'], 
            history['test_loss'] if len(history['test_loss']) == len(history['train_loss']) else [None] * len(history['train_loss']),
            history['cer'] if len(history['cer']) == len(history['train_loss']) else [None] * len(history['train_loss']),
            history['wer'] if len(history['wer']) == len(history['train_loss']) else [None] * len(history['train_loss']),
            history['learning_rate']
        )):
            summary_data.append([i+1, tl, te if te is not None else '-', 
                               c if c is not None else '-', 
                               w if w is not None else '-', lr])
        
        # Create wandb table
        table = wandb.Table(
            columns=["Epoch", "Train Loss", "Test Loss", "CER", "WER", "Learning Rate"],
            data=summary_data
        )
        wandb.log({"training_summary_table": table})
        
        wandb.log({
            'final/test_loss': test_loss,
            'final/cer': cer,
            'final/wer': wer,
        })
    
    # Save final model
    save_path = output_dir / 'final_model.pt'
    save_checkpoint(
        model, optimizer, scheduler, num_epochs - 1, global_step, best_cer, str(save_path)
    )
    
    # Save final model to wandb
    if use_wandb:
        # Try to save model file (may fail on Windows without admin privileges)
        try:
            wandb.save(str(save_path), policy='now')
        except (OSError, PermissionError) as e:
            print(f"Note: Could not upload final model to wandb (Windows symlink restriction)")
            print(f"Model saved locally at: {save_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
    
    # Log training time and GPU stats
    if use_wandb:
        wandb.run.summary['training_time_hours'] = elapsed_time / 3600
        if torch.cuda.is_available():
            wandb.run.summary['gpu_max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
            wandb.run.summary['gpu_max_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
        wandb.finish()
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speech BCI neural decoder")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (for smoke testing)"
    )
    
    args = parser.parse_args()
    
    train(args.config, args.resume, args.epochs)

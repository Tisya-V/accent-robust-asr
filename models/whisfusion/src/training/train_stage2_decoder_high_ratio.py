# -*- coding: utf-8 -*-
"""
train_stage2_decoder_high_ratio.py

Stage 2 training for Whisfusion - fine-tunes full decoder and adapter with high masking ratios
to specialize in initial token generation.

"""


# Add project root to path
import sys
from pathlib import Path
import shutil
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"Adding project root to path: {PROJECT_ROOT}")
print(f"Adding src root to path: {SRC_ROOT}")

import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import time
import math
from typing import Optional, Tuple, Union, Dict
import os
import jiwer
from torch.utils.tensorboard import SummaryWriter

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data.distributed import DistributedSampler

# EMA library
try:
   from torch_ema import ExponentialMovingAverage
   EMA_AVAILABLE = True
except ImportError:
   EMA_AVAILABLE = False
   print("Warning: torch-ema not installed. EMA will be disabled.")


from src.lit_gpt.diffmodel import TransEncoder, Block, Config
from data.dataset_stage1 import create_stage1_dataloader
from safetensors.torch import load_file
from transformers import AutoTokenizer

def forward_process(batch, mask_token_id: int, mask_range=(0.7, 1.0), eps=1e-3):
   """Randomly masks portions of input batch with configurable mask ratio range."""
   b, l = batch.shape
   
   # Mask ratio range (default: 0.7 ~ 1.0 for high masking)
   min_mask, max_mask = mask_range
   
   # Generate random masking ratio within specified range
   t = torch.rand((b,), device=batch.device)
   p_mask = min_mask + (max_mask - min_mask) * t
   
   # Safety clamp with eps
   p_mask = torch.clamp(p_mask, min=eps, max=1-eps)
   
   p_mask = p_mask[:, None].repeat(1, l)
   mask_indices = torch.rand((b, l), device=batch.device) < p_mask
   noisy_batch = torch.where(mask_indices, mask_token_id, batch)
   return noisy_batch, mask_indices

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, 
            config: Config, val_steps: int, tokenizer=None, compute_wer_cer: bool = True,
            ema=None, mask_range=(0.7, 1.0)) -> Dict[str, float]:
   """Evaluates model performance on validation dataset."""
   if fabric.global_rank == 0:
       fabric.print("\nValidating...")
   
   model.eval()
   losses = []
   wer_scores = []
   cer_scores = []
   loss_func = torch.nn.CrossEntropyLoss()
   
   # Use EMA context manager if available
   context_manager = ema.average_parameters() if ema is not None else torch.no_grad()
   
   with context_manager:
       # Ensure consistent validation steps across all processes
       total_val_batches = len(val_dataloader)
       if val_steps > 0:
           total_steps = min(val_steps, total_val_batches)
       else:
           total_steps = total_val_batches
       
       total_steps = fabric.broadcast(torch.tensor(total_steps, device=fabric.device), src=0).item()
       
       # Limit WER/CER samples for efficiency
       max_wer_samples = min(100, total_steps * val_dataloader.batch_size)
       wer_sample_count = 0
       
       processed_batches = 0
       for i, batch in enumerate(val_dataloader):
           if i >= total_steps:
               break
           
           condition = batch['condition']
           target_ids = batch['target_ids']
           
           # Convert BF16 to FP32 if needed
           if condition.dtype == torch.bfloat16:
               condition = condition.float()

           # Use mask_range parameter
           noisy_input, mask_indices = forward_process(
               target_ids, 
               mask_token_id=config.padded_vocab_size,
               mask_range=mask_range
           )
           
           logits = model(idx=noisy_input, condition=condition)
           loss = loss_func(logits[mask_indices], target_ids[mask_indices])
           losses.append(loss.item())
           
           # Compute WER/CER if enabled
           if compute_wer_cer and tokenizer is not None and wer_sample_count < max_wer_samples:
               predicted_ids = torch.argmax(logits, dim=-1)
               reconstructed = torch.where(mask_indices, predicted_ids, noisy_input)
               
               batch_size = min(target_ids.size(0), max_wer_samples - wer_sample_count)
               if batch_size > 0:
                   # Use batch_decode for efficiency
                   target_texts = tokenizer.batch_decode(target_ids[:batch_size], skip_special_tokens=True)
                   pred_texts = tokenizer.batch_decode(reconstructed[:batch_size], skip_special_tokens=True)
                   
                   for target_text, pred_text in zip(target_texts, pred_texts):
                       target_text = target_text.strip()
                       pred_text = pred_text.strip()
                       
                       if target_text and pred_text:
                           try:
                               wer = jiwer.wer(target_text, pred_text)
                               cer = jiwer.cer(target_text, pred_text)
                               wer_scores.append(wer)
                               cer_scores.append(cer)
                               wer_sample_count += 1
                           except:
                               pass
           
           processed_batches += 1
   
   # Calculate local averages
   if processed_batches > 0:
       local_loss = sum(losses) / processed_batches
   else:
       local_loss = 0.0
   
   local_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
   local_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
   local_wer_count = len(wer_scores)
   
   # Gather metrics across all processes
   metrics_tensor = torch.tensor([
       processed_batches, 
       local_loss * processed_batches,
       local_wer * local_wer_count,
       local_cer * local_wer_count,
       local_wer_count
   ], device=fabric.device)
   
   total_metrics = fabric.all_reduce(metrics_tensor, reduce_op="sum")
   
   # Calculate global averages
   total_batches = int(total_metrics[0].item())
   total_wer_samples = int(total_metrics[4].item())
   
   if total_batches > 0:
       avg_loss = total_metrics[1].item() / total_batches
   else:
       avg_loss = 0.0
       
   if total_wer_samples > 0:
       avg_wer = total_metrics[2].item() / total_wer_samples
       avg_cer = total_metrics[3].item() / total_wer_samples
   else:
       avg_wer = 0.0
       avg_cer = 0.0
   
   model.train()
   
   return {
       'loss': avg_loss,
       'wer': avg_wer,
       'cer': avg_cer
   }


def create_fabric_dataloader(fabric: L.Fabric, data_dir: str, tokenizer_name: str, 
                          batch_size: int, num_workers: int, shuffle: bool = True):
   """Creates dataloader for use with Lightning Fabric."""
   dataloader = create_stage1_dataloader(
       data_dir=data_dir,
       tokenizer_name=tokenizer_name,
       batch_size=batch_size,
       num_workers=num_workers,
       shuffle=shuffle
   )
   
   return fabric.setup_dataloaders(dataloader)

def setup(args):
   """Sets up Lightning Fabric and starts training."""
   out_dir = Path(args.out_dir)
   
   if not args.resume:
       if not out_dir.name.startswith(f"ft2-{args.model_name}-"):
           out_dir = out_dir / f"ft2-{args.model_name}-{int(time.time())}"
   
   # Configure Fabric strategy
   if args.num_devices > 1:
       strategy = "ddp"
   else:
       strategy = "auto"
   
   fabric = L.Fabric(
       devices=args.num_devices,
       accelerator="gpu",
       strategy=strategy,
       precision=args.precision
   )
   
   main(fabric, args, out_dir)

def main(fabric: L.Fabric, args, out_dir: Path):
   """Main training function for Stage 2 fine-tuning."""
   
   log_file_path = out_dir / "training_log.txt"
   
   # Initialize TensorBoard writer (rank 0 only)
   if fabric.global_rank == 0:
       writer = SummaryWriter(log_dir=out_dir / "tensorboard")
   else:
       writer = None
   
   # Only rank 0 creates directories and logs
   if fabric.global_rank == 0:
       out_dir.mkdir(parents=True, exist_ok=True)
       fabric.print(f"Stage 2 fine-tuning - Output directory: {out_dir}")
       fabric.print("==> Training full model weights (Full fine-tuning)")
   
   fabric.barrier()
   
   # Set random seed
   fabric.seed_everything(42)
   
   # Initialize tokenizer for WER/CER calculation
   tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
   
   # Prepare dataloaders
   device_batch_size = args.batch_size // fabric.world_size
   val_batch_size = max(16, device_batch_size // 2)

   # Adjust workers for multi-GPU
   adjusted_num_workers = max(1, args.num_workers // fabric.world_size)

   train_dataloader = create_fabric_dataloader(
       fabric, args.train_data_dir, args.tokenizer_name, 
       device_batch_size, adjusted_num_workers, shuffle=True
   )
   val_dataloader = create_fabric_dataloader(
       fabric, args.val_data_dir, args.tokenizer_name,
       val_batch_size,
       adjusted_num_workers, shuffle=False
   )

   # Initialize model
   config = Config.from_name(args.model_name)
   
   with fabric.init_module(empty_init=False):
       model = TransEncoder(config)
   
   # Load pretrained weights
   if args.pretrain_path:
       try:
           if args.pretrain_path.endswith('stage1_adapter.pt'):
               # Load adapter weights from Stage 1
               if fabric.global_rank == 0:
                   fabric.print(f"Loading adapter weights from Stage 1: '{args.pretrain_path}'")
               
               # Load on rank 0 only
               if fabric.global_rank == 0:
                   base_state    = load_file(args.base_model_path, device="cpu")
                   adapter_state = torch.load(args.pretrain_path, map_location="cpu", weights_only=True)
               else:
                   base_state = adapter_state = None

               # Broadcast to all ranks
               base_state, adapter_state = fabric.broadcast([base_state, adapter_state], src=0)

               # Apply weights
               model.load_state_dict(base_state, strict=False)
               model_state = model.state_dict()
               for k, v in adapter_state.items():
                   if k in model_state:
                       model_state[k] = v
               model.load_state_dict(model_state, strict=False)

               if fabric.global_rank == 0:
                   fabric.print(f"Successfully loaded {len(adapter_state)} adapter parameters")
                   fabric.print("Now unfreezing ALL model parameters for Stage 2 fine-tuning")
                   
           else:
               # Load full model
               if fabric.global_rank == 0:
                   state_dict = load_file(args.pretrain_path, device="cpu")
               else:
                   state_dict = None
               
               state_dict, = fabric.broadcast_object_list([state_dict], src=0)
               
               model.load_state_dict(state_dict, strict=False)
               if fabric.global_rank == 0:
                   fabric.print(f"Loaded full model weights from '{args.pretrain_path}'")
                   
       except Exception as e:
           if fabric.global_rank == 0:
               fabric.print(f"Failed to load weights: {e}")
               fabric.print(f"File path: {args.pretrain_path}")
               fabric.print(f"File exists: {os.path.exists(args.pretrain_path)}")
           raise
   
   # Apply learning rate scaling for Stage 2
   effective_batch_size = args.batch_size
   base_batch_size = 64
   
   if args.lr_scaling == 'linear':
       lr_scale = effective_batch_size / base_batch_size
   elif args.lr_scaling == 'sqrt':
       lr_scale = math.sqrt(effective_batch_size / base_batch_size)
   else:
       lr_scale = 1.0
   
   # Stage 2 uses lower learning rate
   scaled_lr = args.learning_rate * lr_scale * args.second_stage_lr_multiplier
   
   # Learning rate cap
   scaled_lr = min(scaled_lr, 1e-4)
   
   if fabric.global_rank == 0:
       fabric.print(f"Learning rate scaling: {args.learning_rate} -> {scaled_lr}")
       fabric.print(f"  (scale: {lr_scale:.2f}, Stage 2 multiplier: {args.second_stage_lr_multiplier})")
       if scaled_lr == 1e-4:
           fabric.print(f"  Learning rate capped at 1e-4")
   
   # Setup optimizer with weight decay groups
   no_decay = ['bias', 'LayerNorm', 'norm']
   
   if args.use_layer_wise_lr_decay:
       # Layer-wise learning rate decay
       if fabric.global_rank == 0:
           fabric.print("Applying layer-wise learning rate decay with weight decay exceptions")
       
       parameter_groups = []
       lr_decay_rate = args.layer_wise_lr_decay_rate
       
       # Cross attention layers (highest learning rate)
       cross_attn_params = {'decay': [], 'no_decay': []}
       # Transformer blocks (layer-wise decay)
       block_params = {i: {'decay': [], 'no_decay': []} for i in range(config.n_layer)}
       # Other parameters
       other_params = {'decay': [], 'no_decay': []}
       
       for name, param in model.named_parameters():
           param.requires_grad = True  # All parameters trainable
           
           # Determine weight decay
           if any(nd in name for nd in no_decay):
               param_type = 'no_decay'
           else:
               param_type = 'decay'
           
           if 'cross_attn' in name or 'norm_cross' in name:
               cross_attn_params[param_type].append(param)
           elif 'transformer.h.' in name:
               # Extract block number
               block_idx = int(name.split('.')[2])
               block_params[block_idx][param_type].append(param)
           else:
               other_params[param_type].append(param)
       
       # Cross attention uses base learning rate
       for param_type, params in cross_attn_params.items():
           if params:
               parameter_groups.append({
                   'params': params,
                   'lr': scaled_lr,
                   'weight_decay': args.weight_decay if param_type == 'decay' else 0.0,
                   'name': f'cross_attention_{param_type}'
               })
       
       # Apply decay for each block (deeper layers get lower lr)
       for i in range(config.n_layer):
           layer_lr = scaled_lr * (lr_decay_rate ** (config.n_layer - i - 1))
           for param_type, params in block_params[i].items():
               if params:
                   parameter_groups.append({
                       'params': params,
                       'lr': layer_lr,
                       'weight_decay': args.weight_decay if param_type == 'decay' else 0.0,
                       'name': f'block_{i}_{param_type}'
                   })
       
       # Other parameters use lowest learning rate
       lowest_lr = scaled_lr * (lr_decay_rate ** config.n_layer)
       for param_type, params in other_params.items():
           if params:
               parameter_groups.append({
                   'params': params,
                   'lr': lowest_lr,
                   'weight_decay': args.weight_decay if param_type == 'decay' else 0.0,
                   'name': f'other_{param_type}'
               })
   else:
       # Single learning rate with weight decay groups
       parameter_groups = []
       decay_params = []
       no_decay_params = []
       
       for name, param in model.named_parameters():
           param.requires_grad = True
           if any(nd in name for nd in no_decay):
               no_decay_params.append(param)
           else:
               decay_params.append(param)
       
       parameter_groups = [
           {'params': decay_params, 'weight_decay': args.weight_decay},
           {'params': no_decay_params, 'weight_decay': 0.0}
       ]
   
   # Log parameter info (rank 0 only)
   if fabric.global_rank == 0:
       fabric.print("\n--- Stage 2 Fine-tuning Parameter Info ---")
       total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       fabric.print(f"Total parameters: {total_params:,}")
       fabric.print(f"Trainable parameters: {trainable_params:,}")
       fabric.print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")
       
       if args.use_layer_wise_lr_decay:
           fabric.print("\nLayer-wise learning rates:")
           printed_groups = set()
           for group in parameter_groups:
               group_name = group['name'].split('_')[0] + '_' + group['name'].split('_')[1]
               if group_name not in printed_groups:
                   fabric.print(f"  {group_name}: {group['lr']:.2e}")
                   printed_groups.add(group_name)
       fabric.print("--------------------------\n")
   
   model = fabric.setup(model)
   
   # Setup optimizer
   optimizer = torch.optim.AdamW(
       parameter_groups, 
       betas=(0.9, 0.999),  # More conservative momentum
       eps=1e-8
   )
   
   optimizer = fabric.setup_optimizers(optimizer)
   
   # Initialize EMA if available
   if args.use_ema and EMA_AVAILABLE:
       ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
       if fabric.global_rank == 0:
           fabric.print(f"Using EMA with decay={args.ema_decay}")
   else:
       ema = None
   
   # Setup scheduler
   steps_per_epoch = len(train_dataloader)
   effective_steps_per_epoch = steps_per_epoch // args.gradient_accumulation_steps
   total_effective_steps = args.epochs * effective_steps_per_epoch
   
   # Stage 2 uses smoother scheduling
   if args.scheduler_type == "cosine":
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=total_effective_steps, eta_min=scaled_lr/100
       )
       scheduler_step_on_batch = True
   elif args.scheduler_type == "constant_with_warmup":
       # Constant lr after warmup
       warmup_steps = int(args.warmup_ratio * total_effective_steps)
       
       def lr_lambda(step):
           if step < warmup_steps:
               return step / warmup_steps
           return 1.0
       
       scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
       scheduler_step_on_batch = True
   else:  # "cosine_epoch"
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=args.epochs, eta_min=scaled_lr/100
       )
       scheduler_step_on_batch = False
   
   # Add warmup (important for Stage 2)
   if args.warmup_ratio > 0:
       if scheduler_step_on_batch:
           warmup_steps = int(args.warmup_ratio * total_effective_steps)
       else:
           warmup_steps = int(args.warmup_ratio * args.epochs)
       
       # Linear warmup
       warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
           optimizer, 
           start_factor=0.01,  # Very low starting lr
           total_iters=warmup_steps
       )
       
       # Main scheduler after warmup
       if args.scheduler_type == "cosine":
           main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
               optimizer, 
               T_max=total_effective_steps - warmup_steps, 
               eta_min=scaled_lr/100
           )
       elif args.scheduler_type == "constant_with_warmup":
           main_scheduler = torch.optim.lr_scheduler.ConstantLR(
               optimizer, factor=1.0
           )
       else:
           main_scheduler = scheduler
       
       # Combine schedulers
       scheduler = torch.optim.lr_scheduler.SequentialLR(
           optimizer,
           schedulers=[warmup_scheduler, main_scheduler],
           milestones=[warmup_steps]
       )
       
       if fabric.global_rank == 0:
           warmup_type = "steps" if scheduler_step_on_batch else "epochs"
           fabric.print(f"Using extended warmup for {warmup_steps} {warmup_type} ({args.warmup_ratio*100:.0f}% of training)")
   
   loss_func = torch.nn.CrossEntropyLoss()
   
   # State dict for checkpointing
   state = {
       "model": model,
       "optimizer": optimizer,
       "scheduler": scheduler,
       "epoch": 0,
       "best_val_loss": float('inf'),
       "best_val_wer": float('inf'),
       "patience_counter": 0
   }
   
   if ema is not None:
       state["ema"] = ema
   
   # Resume from checkpoint if specified
   if args.resume and (out_dir / "last.ckpt").exists():
       try:
           fabric.load(out_dir / "last.ckpt", state)
           if fabric.global_rank == 0:
               fabric.print(f"Resumed from checkpoint: {out_dir / 'last.ckpt'}")
       except Exception as e:
           if fabric.global_rank == 0:
               fabric.print(f"Failed to load checkpoint: {e}")
           raise
   
   # Training loop
   if fabric.global_rank == 0:
       fabric.print("Starting Stage 2 fine-tuning...")
       fabric.print(f"Total epochs: {args.epochs}")
       fabric.print(f"Steps per epoch: {steps_per_epoch}")
       fabric.print(f"Effective steps (with gradient accumulation): {effective_steps_per_epoch}")
       fabric.print(f"Total effective steps: {total_effective_steps}")
       fabric.print(f"Device batch size: {device_batch_size}")
       fabric.print(f"Total batch size: {args.batch_size}")
       fabric.print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
       fabric.print(f"Base learning rate: {args.learning_rate}")
       fabric.print(f"Scaled learning rate: {scaled_lr}")
       fabric.print(f"Weight decay: {args.weight_decay}")
       fabric.print(f"Gradient clipping: {args.gradient_clip_val}")
   
   # Clear GPU memory
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       
   for epoch in range(state["epoch"], args.epochs):
       if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
           train_dataloader.sampler.set_epoch(epoch)
       if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'set_epoch'):
           val_dataloader.sampler.set_epoch(epoch)
           
       model.train()
       
       fabric.barrier()
       
       all_losses = []
       log_interval = max(1, steps_per_epoch // 10)
       
       if fabric.global_rank == 0:
           progress_bar = tqdm(enumerate(train_dataloader), total=steps_per_epoch, 
                           desc=f"Epoch {epoch+1}/{args.epochs}")
       else:
           progress_bar = enumerate(train_dataloader)
           
       accumulation_steps = args.gradient_accumulation_steps

       for batch_idx, batch in progress_bar:
           condition = batch['condition']
           target_ids = batch['target_ids']
           
           # Convert BF16 to FP32 if needed
           if condition.dtype == torch.bfloat16:
               condition = condition.float()
           
           noisy_input, mask_indices = forward_process(
               target_ids, 
               mask_token_id=config.padded_vocab_size,
               mask_range=(args.min_mask_ratio, args.max_mask_ratio)  # Use command line args
           )
           
           logits = model(idx=noisy_input, condition=condition)
           loss = loss_func(logits[mask_indices], target_ids[mask_indices])
           
           # Scale loss for gradient accumulation
           loss = loss / accumulation_steps
           
           fabric.backward(loss)
           
           # Update weights after accumulation steps
           if (batch_idx + 1) % accumulation_steps == 0:
               # Conservative gradient clipping for Stage 2
               fabric.clip_gradients(model, optimizer, max_norm=args.gradient_clip_val)
               
               optimizer.step()
               optimizer.zero_grad()
               
               # Update EMA
               if ema is not None:
                   ema.update()
               
               if scheduler_step_on_batch:
                   scheduler.step()
               
           # Record loss
           all_losses.append(loss.item() * accumulation_steps)
           
           # Periodic logging
           if (batch_idx + 1) % log_interval == 0:
               recent_losses = all_losses[-log_interval:]
               avg_loss = sum(recent_losses) / len(recent_losses)
               current_lr = scheduler.get_last_lr()[0]
               
               if fabric.global_rank == 0 and hasattr(progress_bar, 'set_postfix'):
                   progress_bar.set_postfix(loss=avg_loss, lr=current_lr)
       
       # Update scheduler per epoch
       if not scheduler_step_on_batch:
           scheduler.step()
       
       fabric.barrier()
       
       # Calculate epoch average loss
       epoch_avg_loss = None
       if fabric.global_rank == 0 and all_losses:
           epoch_avg_loss = sum(all_losses) / len(all_losses)
           fabric.print(f"\nEpoch {epoch+1} - Average training loss: {epoch_avg_loss:.4f}")

       # Validation
       val_metrics = validate(
           fabric, model, val_dataloader, config, args.val_steps, 
           tokenizer=tokenizer, 
           compute_wer_cer=args.compute_wer_cer,
           ema=ema, 
           mask_range=(args.min_mask_ratio, args.max_mask_ratio)
       )
       val_loss = val_metrics['loss']
       val_wer = val_metrics['wer']
       val_cer = val_metrics['cer']
       
       # TensorBoard logging (rank 0 only)
       if fabric.global_rank == 0 and writer is not None:
           writer.add_scalar('Loss/train', epoch_avg_loss, epoch)
           writer.add_scalar('Loss/val', val_loss, epoch)
           writer.add_scalar('Metrics/WER', val_wer, epoch)
           writer.add_scalar('Metrics/CER', val_cer, epoch)
           writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
       
       fabric.barrier()
       
       # Initialize tensors for broadcast
       should_stop = torch.tensor(False, device=fabric.device)
       save_best = torch.tensor(False, device=fabric.device)
       
       # Logging and early stopping logic (rank 0 only)
       if fabric.global_rank == 0:
           current_lr = scheduler.get_last_lr()[0]
           log_message = f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}"
           
           if args.compute_wer_cer:
               log_message += f" | WER: {val_wer:.4f} | CER: {val_cer:.4f}"
           
           log_message += f" | LR: {current_lr:.2e}"
           
           if epoch_avg_loss is not None:
               log_message = f"Epoch {epoch+1} | Train Loss: {epoch_avg_loss:.4f} | " + log_message.split(" | ", 1)[1]
           
           fabric.print(log_message)
           
           # Check for best model
           if args.early_stop_metric == 'wer' and args.compute_wer_cer:
               current_metric = val_wer
               best_metric = state["best_val_wer"]
               metric_name = "WER"
           else:
               current_metric = val_loss
               best_metric = state["best_val_loss"]
               metric_name = "validation loss"
           
           if current_metric < best_metric:
               if args.early_stop_metric == 'wer':
                   state["best_val_wer"] = current_metric
               else:
                   state["best_val_loss"] = current_metric
               state["patience_counter"] = 0
               save_best = torch.tensor(True, device=fabric.device)
               
               save_message = f" -> {metric_name} improved."
               fabric.print(save_message)
               log_message += save_message
           else:
               state["patience_counter"] += 1
               patience_message = f" -> {metric_name} did not improve. Patience: {state['patience_counter']}/{args.patience}"
               fabric.print(patience_message)
               log_message += patience_message
           
           # Write to log file
           with open(log_file_path, "a") as f:
               f.write(log_message + "\n")
           
           # Check early stopping
           if state["patience_counter"] >= args.patience:
               should_stop = torch.tensor(True, device=fabric.device)
               early_stop_message = f"Early stopping triggered after {args.patience} epochs with no improvement."
               fabric.print(early_stop_message)
               with open(log_file_path, "a") as f:
                   f.write(early_stop_message + "\n")
       
       # Broadcast decisions to all processes
       save_best = fabric.broadcast(save_best, src=0)
       should_stop = fabric.broadcast(should_stop, src=0)
       
       # Save best model (full model for Stage 2)
       if save_best.item():
           save_path = out_dir / "model_best.pt"
           fabric.save(save_path, model.state_dict())
           if fabric.global_rank == 0:
               fabric.print(f"Saved best full model to {save_path}")
           
           # Save EMA model if using
           if ema is not None:
               ema_save_path = out_dir / "model_best_ema.pt"
               with ema.average_parameters():
                   fabric.save(ema_save_path, model.state_dict())
               if fabric.global_rank == 0:
                   fabric.print(f"Saved best EMA model to {ema_save_path}")
       
       # Save last checkpoint
       state["epoch"] = epoch + 1
       fabric.save(out_dir / "last.ckpt", state)
       
       # Periodic checkpoint saving
       if (epoch + 1) % args.save_every_n_epochs == 0:
           epoch_save_path = out_dir / f"checkpoint_epoch_{epoch+1}.ckpt"
           fabric.save(epoch_save_path, state)
           if fabric.global_rank == 0:
               fabric.print(f"Saved checkpoint at epoch {epoch+1}")
       
       fabric.barrier()
       
       # Clear GPU memory
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
       
       if should_stop.item():
           if fabric.global_rank == 0:
               fabric.print("==> Early stop, breaking training loop")
           break
   
   if fabric.global_rank == 0:
       fabric.print("\nStage 2 fine-tuning complete!")
       if args.early_stop_metric == 'wer' and 'best_val_wer' in state:
           fabric.print(f"Best validation WER: {state['best_val_wer']:.4f}")
       else:
           fabric.print(f"Best validation loss: {state['best_val_loss']:.4f}")

       fabric.print(f"Best model saved to: {out_dir.parent / f'{args.out_model_name}_stage2_decoder.pt'}")
       
       final_save_path = out_dir.parent / f"{args.out_model_name}_stage2_decoder.pt"
       shutil.copy2(out_dir / "model_best.pt", final_save_path)
       
       # Close TensorBoard writer
       if writer is not None:
           writer.close()
       

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Stage 2 Whisfusion fine-tuning with high masking ratios")
   
   # Path arguments
   parser.add_argument('--train_data_dir', type=str, nargs='+', required=True, 
                      help='Training data directories (space-separated)')
   parser.add_argument('--val_data_dir', type=str, nargs='+', required=True, 
                      help='Validation data directories (space-separated)')
   parser.add_argument('--pretrain_path', type=str, required=True,
                      help='Stage 1 adapter_best.pt or full model path')
   parser.add_argument('--base_model_path', type=str, 
                      default="pretrained_models/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors")
   parser.add_argument('--out_dir', type=str, default="output/finetune_stage2_decoder")
   parser.add_argument('--out_model_name', type=str, default="whisfusion", help='Base name for output model files')

   # Model and tokenizer
   parser.add_argument('--model_name', type=str, default="Diff_LLaMA_170M")
   parser.add_argument('--tokenizer_name', type=str, 
                      default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
   
   # Training hyperparameters (Stage 2 defaults)
   parser.add_argument('--batch_size', type=int, default=32, 
                      help='Total batch size across all GPUs')
   parser.add_argument('--epochs', type=int, default=20)
   parser.add_argument('--learning_rate', type=float, default=1e-5)
   parser.add_argument('--second_stage_lr_multiplier', type=float, default=0.1,
                      help='Additional lr reduction factor for Stage 2')
   parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for regularization')
   parser.add_argument('--gradient_clip_val', type=float, default=0.5,
                      help='Gradient clipping value')
   parser.add_argument('--scheduler_type', type=str, default="cosine", 
                      choices=["cosine", "cosine_epoch", "constant_with_warmup"], 
                      help="Scheduler type")
   
   # Layer-wise learning rate decay
   parser.add_argument('--use_layer_wise_lr_decay', action='store_true',
                      help='Apply different learning rates per layer')
   parser.add_argument('--layer_wise_lr_decay_rate', type=float, default=0.9,
                      help='Layer-wise LR decay rate')
   
   # Learning rate scaling
   parser.add_argument('--lr_scaling', type=str, default='sqrt', 
                      choices=['none', 'linear', 'sqrt'],
                      help='Multi-GPU learning rate scaling method')
   parser.add_argument('--warmup_ratio', type=float, default=0.2,
                      help='Warmup ratio')
   
   # EMA
   parser.add_argument('--use_ema', action='store_true',
                      help='Use Exponential Moving Average of model weights')
   parser.add_argument('--ema_decay', type=float, default=0.995,
                      help='EMA decay rate')
   
   # Validation and early stopping
   parser.add_argument('--val_steps', type=int, default=-1)
   parser.add_argument('--patience', type=int, default=5)
   parser.add_argument('--compute_wer_cer', action='store_true')
   parser.add_argument('--early_stop_metric', type=str, default='loss',
                      choices=['loss', 'wer'], help='Early stopping metric')
   parser.add_argument('--save_every_n_epochs', type=int, default=5,
                      help='Save checkpoint every N epochs')
   
   # Hardware
   parser.add_argument('--num_devices', type=int, default=1, help='Number of GPUs')
   parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')
   parser.add_argument('--precision', type=str, default="32-true", 
                      choices=["32-true", "16-mixed", "bf16-mixed"])
   
   parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                  help='Gradient accumulation steps')
   
   # Masking ratios
   parser.add_argument('--min_mask_ratio', type=float, default=0.7,
                   help='Minimum masking ratio for high masking')
   parser.add_argument('--max_mask_ratio', type=float, default=1.0,
                   help='Maximum masking ratio for high masking')

   # Resume training
   parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
   
   args = parser.parse_args()
   
   # PyTorch settings
   torch.set_float32_matmul_precision('high')
   
   # Enable TF32 for Ampere GPUs
   if torch.cuda.is_available():
       torch.backends.cuda.matmul.allow_tf32 = True
       torch.backends.cudnn.allow_tf32 = True
   
   # NCCL timeout for multi-GPU
   if args.num_devices > 1:
       os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes
   
   setup(args)
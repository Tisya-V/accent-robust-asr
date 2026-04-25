# -*- coding: utf-8 -*-
"""
train_stage1_adapter.py

Stage 1 training for Whisfusion - trains only cross-attention adapter layers
while keeping the rest of the model frozen.

"""
# Add project root to path
import shutil
import sys
from pathlib import Path
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

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data.distributed import DistributedSampler


from src.lit_gpt.diffmodel import TransEncoder, Block, Config
from data.dataset_stage1 import create_stage1_dataloader
from safetensors.torch import load_file
from transformers import AutoTokenizer

def forward_process(batch, mask_token_id: int, eps=1e-3):
   """Randomly masks portions of input batch for denoising training."""
   b, l = batch.shape
   t = torch.rand((b,), device=batch.device)
   p_mask = (1 - eps) * t + eps
   p_mask = p_mask[:, None].repeat(1, l)
   mask_indices = torch.rand((b, l), device=batch.device) < p_mask
   noisy_batch = torch.where(mask_indices, mask_token_id, batch)
   return noisy_batch, mask_indices


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, 
            config: Config, val_steps: int, tokenizer=None, compute_wer_cer: bool = True) -> Dict[str, float]:
   """Evaluates model performance on validation dataset."""
   if fabric.global_rank == 0:
       fabric.print("\nValidating...")
   
   model.eval()
   losses = []
   wer_scores = []
   cer_scores = []
   loss_func = torch.nn.CrossEntropyLoss()
   
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

       noisy_input, mask_indices = forward_process(
           target_ids, mask_token_id=config.padded_vocab_size
       )
       
       logits = model(idx=noisy_input, condition=condition)
       loss = loss_func(logits[mask_indices], target_ids[mask_indices])
       losses.append(loss.item())
       
       # Compute WER/CER if enabled
       if compute_wer_cer and tokenizer is not None and wer_sample_count < max_wer_samples:
           predicted_ids = torch.argmax(logits, dim=-1)
           reconstructed = torch.where(mask_indices, predicted_ids, noisy_input)
           
           batch_size = min(target_ids.size(0), max_wer_samples - wer_sample_count)
           for j in range(batch_size):
               target_text = tokenizer.decode(target_ids[j], skip_special_tokens=True).strip()
               pred_text = tokenizer.decode(reconstructed[j], skip_special_tokens=True).strip()
               
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
       if not out_dir.name.startswith(f"ft-{args.model_name}-"):
           out_dir = out_dir / f"ft-{args.model_name}-{int(time.time())}"
   
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
   """Main training function."""
   log_file_path = out_dir / "training_log.txt"
   
   # Only rank 0 creates directories and logs
   if fabric.global_rank == 0:
       out_dir.mkdir(parents=True, exist_ok=True)
       fabric.print(f"Checkpoint and log directory: {out_dir}")
   
   fabric.barrier()
   
   # Set random seed
   fabric.seed_everything(42)
   
   # Initialize tokenizer for WER/CER calculation
   tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
   
   # Prepare dataloaders
   if fabric.global_rank == 0:
        fabric.print("Preparing dataloaders...")
   device_batch_size = args.batch_size // fabric.world_size
   val_batch_size = max(8, device_batch_size // 2)

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
   if fabric.global_rank == 0:
        fabric.print("Initializing model...")
   config = Config.from_name(args.model_name)
   
   with fabric.init_module(empty_init=False):
       model = TransEncoder(config)
   
   # Load pretrained weights
   if args.pretrain_path:
       try:
           if args.pretrain_path.endswith('adapter_best.pt'):
               # Load only adapter weights
               if fabric.global_rank == 0:
                   fabric.print(f"Loading adapter weights from '{args.pretrain_path}'")
               
               adapter_state = torch.load(args.pretrain_path, map_location="cpu")
               model_state = model.state_dict()
               
               for key, value in adapter_state.items():
                   if key in model_state:
                       model_state[key] = value
                       
               model.load_state_dict(model_state, strict=False)
               
               if fabric.global_rank == 0:
                   fabric.print(f"Successfully loaded {len(adapter_state)} adapter parameters")
           else:
               # Load full model from safetensors
               state_dict = load_file(args.pretrain_path, device="cpu")
               model.load_state_dict(state_dict, strict=False)
               if fabric.global_rank == 0:
                   fabric.print(f"Loaded pretrained weights from '{args.pretrain_path}'")
                   
       except Exception as e:
           if fabric.global_rank == 0:
               fabric.print(f"Failed to load weights: {e}")
               fabric.print(f"File path: {args.pretrain_path}")
               fabric.print(f"File exists: {os.path.exists(args.pretrain_path)}")
           raise
   
   # Freeze all parameters except cross-attention layers
   for name, param in model.named_parameters():
       if 'cross_attn' not in name and 'norm_cross' not in name:
           param.requires_grad = False
   
   if fabric.global_rank == 0:
       fabric.print("\n--- Trainable Parameters ---")
       trainable_params_count = 0
       for name, param in model.named_parameters():
           if param.requires_grad:
               fabric.print(name)
               trainable_params_count += param.numel()
       fabric.print(f"--------------------------")
       fabric.print(f"Total trainable parameters: {trainable_params_count:,}")
       fabric.print("--------------------------\n")
   
   model = fabric.setup(model)
   
   # Apply learning rate scaling
   effective_batch_size = args.batch_size
   base_batch_size = 64
   
   if args.lr_scaling == 'linear':
       lr_scale = effective_batch_size / base_batch_size
   elif args.lr_scaling == 'sqrt':
       lr_scale = math.sqrt(effective_batch_size / base_batch_size)
   else:
       lr_scale = 1.0
   
   scaled_lr = args.learning_rate * lr_scale
   scaled_lr = min(scaled_lr, args.lr_max)  # Apply learning rate cap
   
   if fabric.global_rank == 0:
       fabric.print(f"Learning rate scaling: {args.learning_rate} -> {scaled_lr} (scale: {lr_scale:.2f})")
       if scaled_lr == args.lr_max:
           fabric.print(f"Learning rate capped at {args.lr_max}")
   
   # Setup optimizer with weight decay groups
   no_decay = ['norm', 'bias']
   optimizer_grouped_parameters = [
       {
           'params': [p for n, p in model.named_parameters() 
                     if p.requires_grad and not any(nd in n for nd in no_decay)],
           'weight_decay': args.weight_decay,
       },
       {
           'params': [p for n, p in model.named_parameters() 
                     if p.requires_grad and any(nd in n for nd in no_decay)],
           'weight_decay': 0.0,
       }
   ]
   
   optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=scaled_lr, betas=(0.9, 0.95))
   optimizer = fabric.setup_optimizers(optimizer)
   
   # Setup scheduler
   steps_per_epoch = len(train_dataloader)
   effective_steps_per_epoch = steps_per_epoch // args.gradient_accumulation_steps
   total_effective_steps = args.epochs * effective_steps_per_epoch

   if args.scheduler_type == "cosine":
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=total_effective_steps, eta_min=scaled_lr/10
       )
       scheduler_step_on_batch = True
   else:  # "cosine_epoch"
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=args.epochs, eta_min=scaled_lr/10
       )
       scheduler_step_on_batch = False
   
   # Add warmup if specified
   if args.warmup_ratio > 0:
       if args.scheduler_type == "cosine":
           warmup_steps = int(args.warmup_ratio * total_effective_steps)
       else:
           warmup_steps = int(args.warmup_ratio * args.epochs)
       
       warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
           optimizer, 
           start_factor=0.1, 
           total_iters=warmup_steps
       )
       
       if args.scheduler_type == "cosine":
           main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
               optimizer, 
               T_max=total_effective_steps - warmup_steps, 
               eta_min=scaled_lr/10
           )
       else:
           main_scheduler = scheduler
       
       scheduler = torch.optim.lr_scheduler.SequentialLR(
           optimizer,
           schedulers=[warmup_scheduler, main_scheduler],
           milestones=[warmup_steps]
       )
       
       if fabric.global_rank == 0:
           warmup_type = "steps" if args.scheduler_type == "cosine" else "epochs"
           fabric.print(f"Using warmup for {warmup_steps} {warmup_type} ({args.warmup_ratio*100:.0f}% of training)")
   
   loss_func = torch.nn.CrossEntropyLoss()
   
   # State dict for checkpointing
   state = {
       "model": model,
       "optimizer": optimizer,
       "scheduler": scheduler,
       "epoch": 0,
       "best_val_loss": float('inf'),
       "patience_counter": 0
   }
   
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
       fabric.print("Starting training...")
       fabric.print(f"Total epochs: {args.epochs}")
       fabric.print(f"Steps per epoch: {steps_per_epoch}")
       fabric.print(f"Effective steps (with gradient accumulation): {effective_steps_per_epoch}")
       fabric.print(f"Total effective steps: {total_effective_steps}")
       fabric.print(f"Device batch size: {device_batch_size}")
       fabric.print(f"Total batch size: {args.batch_size}")
       fabric.print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
       fabric.print(f"Base learning rate: {args.learning_rate}")
       fabric.print(f"Scaled learning rate: {scaled_lr}")
       fabric.print(f"Learning rate scaling: {args.lr_scaling}")
       fabric.print(f"Weight Decay: {args.weight_decay}")
       fabric.print(f"Workers per device: {adjusted_num_workers}")
   
   # Clear GPU memory
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       
   for epoch in range(state["epoch"], args.epochs):
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
               mask_token_id=config.padded_vocab_size
           )
           
           logits = model(idx=noisy_input, condition=condition)
           loss = loss_func(logits[mask_indices], target_ids[mask_indices])
           
           # Scale loss for gradient accumulation
           loss = loss / accumulation_steps
           
           fabric.backward(loss)
           
           # Update weights after accumulation steps
           if (batch_idx + 1) % accumulation_steps == 0:
               fabric.clip_gradients(model, optimizer, max_norm=args.clip_grad_norm)
               
               optimizer.step()
               optimizer.zero_grad()
               
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
       val_metrics = validate(fabric, model, val_dataloader, config, args.val_steps, 
                             tokenizer=tokenizer, compute_wer_cer=args.compute_wer_cer)
       val_loss = val_metrics['loss']
       val_wer = val_metrics['wer']
       val_cer = val_metrics['cer']
       
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
           if val_loss < state["best_val_loss"]:
               state["best_val_loss"] = val_loss
               state["patience_counter"] = 0
               save_best = torch.tensor(True, device=fabric.device)
               
               save_message = f" -> Validation loss improved."
               fabric.print(save_message)
               log_message += save_message
           else:
               state["patience_counter"] += 1
               patience_message = f" -> Validation loss did not improve. Patience: {state['patience_counter']}/{args.patience}"
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
       
       # Save best adapter weights
       if save_best.item():
           save_path = out_dir / "adapter_best.pt"
           adapter_state = {
               k: v for k, v in model.state_dict().items() 
               if 'cross_attn' in k or 'norm_cross' in k
           }
           fabric.save(save_path, adapter_state)
           if fabric.global_rank == 0:
               fabric.print(f"Saved best adapter to {save_path}")
       
       # Save last checkpoint
       state["epoch"] = epoch + 1
       fabric.save(out_dir / "last.ckpt", state)
       
       fabric.barrier()
       
       # Clear GPU memory
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
       
       if should_stop.item():
           if fabric.global_rank == 0:
               fabric.print("==> Early stop, breaking training loop")
           break
   
   if fabric.global_rank == 0:
       fabric.print("\nTraining complete!")
       fabric.print(f"Best validation loss: {state['best_val_loss']:.4f}")

       fabric.print(f"Best adapter saved at: {out_dir.parent / 'stage1_adapter.pt'}")
       
       final_save_path = out_dir.parent / "stage1_adapter.pt"
       shutil.copy2(out_dir / "adapter_best.pt", final_save_path)
       

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Stage 1 adapter training for Whisfusion")
   
   # Path arguments
   parser.add_argument('--train_data_dir', type=str, nargs='+', required=True, 
                      help='Training data directories (space-separated)')
   parser.add_argument('--val_data_dir', type=str, nargs='+', required=True, 
                      help='Validation data directories (space-separated)')
   parser.add_argument('--pretrain_path', type=str, 
                      default="pretrained_models/mdm_safetensors/mdm-170M-100e18-rsl-0.01.safetensors")
   parser.add_argument('--out_dir', type=str, default="whisfusion_/stage1_adapter")
   
   # Model and tokenizer
   parser.add_argument('--model_name', type=str, default="Diff_LLaMA_170M")
   parser.add_argument('--tokenizer_name', type=str, 
                      default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
   
   # Training hyperparameters
   parser.add_argument('--batch_size', type=int, default=64, 
                      help='Total batch size across all GPUs')
   parser.add_argument('--epochs', type=int, default=80)
   parser.add_argument('--learning_rate', type=float, default=1e-4)
   parser.add_argument('--lr_max', type=float, default=3e-4, 
                      help='Maximum learning rate after scaling')
   parser.add_argument('--scheduler_type', type=str, default="cosine_epoch", 
                      choices=["cosine", "cosine_epoch"],
                      help="cosine: per-step update, cosine_epoch: per-epoch update")
   
   # Learning rate scaling
   parser.add_argument('--lr_scaling', type=str, default='linear',
                      choices=['none', 'linear', 'sqrt'],
                      help='Multi-GPU learning rate scaling method')
   parser.add_argument('--warmup_ratio', type=float, default=0.02,
                      help='Warmup ratio (0.02 = 2%)')
   
   # Regularization
   parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay (L2 penalty)')
   parser.add_argument('--clip_grad_norm', type=float, default=0.5,
                      help='Gradient clipping max norm')
   
   # Validation and early stopping
   parser.add_argument('--val_steps', type=int, default=-1)
   parser.add_argument('--patience', type=int, default=8)
   parser.add_argument('--compute_wer_cer', type=bool, default=True, 
                      help='Whether to compute WER/CER during validation')
   
   # Hardware
   parser.add_argument('--num_devices', type=int, default=1, help='Number of GPUs to use')
   parser.add_argument('--num_workers', type=int, default=4, help='Total data loading workers')
   parser.add_argument('--precision', type=str, default="32-true", 
                      choices=["32-true", "16-mixed", "bf16-mixed"])
   
   parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Gradient accumulation steps')
   
   # Resume training
   parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
   
   args = parser.parse_args()
   
   # PyTorch settings
   torch.set_float32_matmul_precision('high')
   
   # NCCL timeout for multi-GPU
   if args.num_devices > 1:
       os.environ["NCCL_TIMEOUT"] = "600"  # 10 minutes
   
   setup(args)
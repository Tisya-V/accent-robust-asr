"""
Exp1 configuration: perturbation strategy + model size + training hyperparams.
Saved alongside each checkpoint for reproducibility.
"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Exp1Config:
    # Perturbation strategy (Stage 1)
    use_perturbation: bool = True
    perturb_prob: float = 0.15           # probability of perturbing a visible token
    perturber_k: int = 10                # k nearest phonemic neighbours
    visible_mask_ratio_low: float = 0.3  # min fraction of tokens to keep visible
    visible_mask_ratio_high: float = 0.7 # max fraction of tokens to keep visible
    include_perturb_in_loss: bool = True # weight perturbed tokens higher in loss

    # Model size (MiniMDM)
    n_layers: int = 4
    n_embd: int = 256
    n_heads: int = 4

    # Training (Stage 1)
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 10
    warmup_steps: int = 100
    max_length: int = 256

    # Data
    data_root: str = "data/processed"
    results_dir: str = "results/experiment1_stage1"

    # Tokenizer / vocab
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B"
    vocab_size: int = 32000

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Exp1Config":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
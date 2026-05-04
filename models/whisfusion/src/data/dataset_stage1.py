import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from transformers import AutoTokenizer
from typing import Union, List

# import io


# --- 1. Dataset class for local files ---
class LibriSpeechStage1Dataset(Dataset):
    """
    Loads preprocessed LibriSpeech .pt files from one or multiple directories.
    """

    def __init__(self, data_dir: Union[str, List[str]]):
        """
        Args:
            data_dir (Union[str, List[str]]): Single directory path or list of paths containing .pt files.
        """
        super().__init__()

        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        self.file_paths = []
        for directory in data_dir:
            path = Path(directory)
            if not path.is_dir():
                print(f"Warning: Directory not found, skipping: {path}")
                continue
            self.file_paths.extend(sorted(list(path.glob("**/*.pt"))))

        if not self.file_paths:
            raise FileNotFoundError(
                f"Could not find any .pt files in the provided directories: {data_dir}"
            )

        print(
            f"Found {len(self.file_paths)} total .pt files from {len(data_dir)} directories."
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_paths[idx]
        # weights_only=True is recommended for security when loading data from untrusted sources.
        data = torch.load(file_path, weights_only=True)
        return data


# --- 2. Collate function for creating batches ---
class WhisfusionCollator:
    """
    A custom collate function to handle batching of samples with fixed-length padding.
    """

    def __init__(self, tokenizer_name: str, max_length: int = 256):
        """
        Args:
            tokenizer_name (str): The Hugging Face name of the tokenizer to use.
            max_length (int): The fixed length to pad all sequences to.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # If the tokenizer doesn't have a pad_token, use the eos_token as a fallback.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: list[dict]) -> dict:
        hidden_states_list = [item["hidden_states"] for item in batch]
        transcript_texts = [item["transcript"] for item in batch]

        # Since hidden_states have a fixed length of 1500, use torch.stack for efficiency.
        stacked_hidden_states = torch.stack(hidden_states_list, dim=0)

        tokenized_output = self.tokenizer(
            transcript_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # 256
        )
        padded_transcripts = tokenized_output.input_ids

        return {"condition": stacked_hidden_states, "target_ids": padded_transcripts}


# --- 3. Main function to create the DataLoader ---
def create_stage1_dataloader(
    data_dir: Union[str, List[str]],
    tokenizer_name: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates and returns a DataLoader for the LibriSpeech .pt dataset.

    Args:
        data_dir (str): The local directory where .pt files are stored.
        tokenizer_name (str): The Hugging Face name of the tokenizer to use.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        DataLoader: A configured PyTorch DataLoader object.
    """
    dataset = LibriSpeechStage1Dataset(data_dir=data_dir)
    collator = WhisfusionCollator(tokenizer_name=tokenizer_name)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return dataloader


# --- for debugging ---
if __name__ == "__main__":
    LOCAL_DATA_DIR = "/root/Whisfusion/Whisfusion_model/data/dev-clean"
    TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    BATCH_SIZE = 16
    NUM_BATCHES_TO_CHECK = 3

    print("--- DataLoader Debugging ---")

    try:
        test_dataloader = create_whisfusion_dataloader(
            data_dir=LOCAL_DATA_DIR,
            tokenizer_name=TOKENIZER_NAME,
            batch_size=BATCH_SIZE,
            num_workers=0,
        )

        print("\n" + "=" * 50)
        print(
            f"DataLoader created successfully. Checking {NUM_BATCHES_TO_CHECK} batches."
        )
        print("=" * 50)

        for i, batch in enumerate(test_dataloader):
            if i >= NUM_BATCHES_TO_CHECK:
                break

            print(f"\n--- Batch # {i+1} info ---")

            # Check the keys of the batch dictionary
            print(f"  ▶ Batch Keys: {batch.keys()}")

            # Check the shape of the 'condition' tensor
            condition_tensor = batch["condition"]
            print(f"  ▶ 'condition' tensor shape: {condition_tensor.shape}")
            print(f"        [Batch Size, Time Axis (1500), Embedding Dimension]")

            # Check the shape of the 'target_ids' tensor
            target_ids_tensor = batch["target_ids"]
            print(f"  ▶ 'target_ids' tensor shape: {target_ids_tensor.shape}")
            print(f"        [Batch Size, Max token length in this batch]")

            # Print details of the first sample in the batch
            print(f"  ▶ Shape of one 'condition' sample: {condition_tensor[0].shape}")
            print(f"  ▶ A slice of one 'condition' sample: {condition_tensor[0]}")

            print(f"  ▶ Shape of one 'target_ids' sample: {target_ids_tensor[0].shape}")
            print(f"  ▶ Values of one 'target_ids' sample: {target_ids_tensor[0]}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
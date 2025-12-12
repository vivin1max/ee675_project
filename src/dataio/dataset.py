import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):

    
    def __init__(self, data: List[Dict], transform=None):

        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d['sentenceDat']) for d in data])
        
        # Flatten all trials across days
        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        
        for day_idx in range(self.n_days):
            day_data = data[day_idx]
            for trial_idx in range(len(day_data['sentenceDat'])):
                self.neural_feats.append(day_data['sentenceDat'][trial_idx])
                self.phone_seqs.append(day_data['phonemes'][trial_idx])
                self.neural_time_bins.append(day_data['sentenceDat'][trial_idx].shape[0])
                self.phone_seq_lens.append(day_data['phoneLens'][trial_idx])
                self.days.append(day_idx)
    
    def __len__(self) -> int:
        return self.n_trials
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)
        
        if self.transform:
            neural_feats = self.transform(neural_feats)
        
        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:

    from torch.nn.utils.rnn import pad_sequence
    
    X, y, X_lens, y_lens, days = zip(*batch)
    
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def load_dataset(
    preprocessed_pkl: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:

    with open(preprocessed_pkl, 'rb') as handle:
        loaded_data = pickle.load(handle)
    
    train_ds = SpeechDataset(loaded_data['train'], transform=None)
    test_ds = SpeechDataset(loaded_data['test'])
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return train_loader, test_loader, loaded_data


def get_dataset_info(preprocessed_pkl: str) -> Dict:

    with open(preprocessed_pkl, 'rb') as handle:
        loaded_data = pickle.load(handle)
    
    # Get example trial to determine dimensions
    example_trial = loaded_data['train'][0]['sentenceDat'][0]
    n_input_features = example_trial.shape[1]
    
    # Get number of unique phonemes
    all_phonemes = []
    for split in ['train', 'test']:
        if split in loaded_data:
            for day_data in loaded_data[split]:
                for phoneme_seq in day_data['phonemes']:
                    all_phonemes.extend(phoneme_seq.tolist())
    
    n_classes = len(set(all_phonemes))
    
    return {
        'n_input_features': n_input_features,
        'n_classes': n_classes,
        'n_train_days': len(loaded_data.get('train', [])),
        'n_test_days': len(loaded_data.get('test', [])),
        'n_holdout_days': len(loaded_data.get('holdout', [])),
    }


if __name__ == "__main__":
    # Test loading a dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <preprocessed_pkl_path>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    
    print("Loading dataset...")
    train_loader, test_loader, loaded_data = load_dataset(pkl_path, batch_size=4)
    
    print("\nDataset info:")
    info = get_dataset_info(pkl_path)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nTesting train loader:")
    X, y, X_len, y_len, days = next(iter(train_loader))
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X_len: {X_len}")
    print(f"  y_len: {y_len}")
    print(f"  days: {days}")

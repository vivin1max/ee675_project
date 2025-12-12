import os
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import re

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from g2p_en import G2p

# Phoneme definitions from the paper
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']  

def phoneToId(p):

    return PHONE_DEF_SIL.index(p)


def load_mat_file(mat_path: str) -> Dict:

    try:
        # Try loading with h5py first (for MATLAB v7.3 files)
        with h5py.File(mat_path, 'r') as f:
            data = {}
            for key in f.keys():
                if key not in ['#refs#', '#subsystem#']:
                    data[key] = f[key][:]
            return data
    except (OSError, KeyError):
        # Fall back to scipy.io for older MATLAB formats
        return sio.loadmat(mat_path)


def extract_session_data(mat_data: Dict) -> Tuple[List[np.ndarray], List[str], List[int]]:

    # Extract neural features - ONLY first 128 channels (area 6v)
    # Combine tx1[0:128] and spikePow[0:128] = 256 features total
    spikePow = mat_data['spikePow']
    tx1 = mat_data['tx1']
    sentenceText = mat_data['sentenceText']
    blockIdx = mat_data['blockIdx'].flatten()
    
    n_trials = len(sentenceText)
    
    neural_features_list = []
    sentence_texts = []
    block_indices = []
    
    for trial_idx in range(n_trials):
        # Extract spike power features (T x 128) - area 6v ONLY
        spike_feats = spikePow[0, trial_idx] if spikePow.shape[0] == 1 else spikePow[trial_idx, 0]
        spike_feats = spike_feats[:, 0:128]  # First 128 channels only
        
        # Extract tx1 features (T x 128) - area 6v ONLY
        tx1_feats = tx1[0, trial_idx] if tx1.shape[0] == 1 else tx1[trial_idx, 0]
        tx1_feats = tx1_feats[:, 0:128]  # First 128 channels only
        
        # Concatenate features: (T x 256) - tx1 + spikePow, both area 6v only
        combined_feats = np.concatenate([tx1_feats, spike_feats], axis=1).astype(np.float32)
        
        # Extract sentence text
        sentence = sentenceText[trial_idx]
        if isinstance(sentence, np.ndarray):
            sentence = ''.join(sentence).strip()
        else:
            sentence = str(sentence).strip()
        
        neural_features_list.append(combined_feats)
        sentence_texts.append(sentence)
        block_indices.append(int(blockIdx[trial_idx]))
    
    return neural_features_list, sentence_texts, block_indices


def process_day_data(day_mat_files: List[str]) -> Dict:

    # Each day should have exactly one .mat file
    if len(day_mat_files) != 1:
        print(f"Warning: Expected 1 file per day, got {len(day_mat_files)}")
    
    mat_file = day_mat_files[0]
    
    try:
        mat_data = load_mat_file(mat_file)
        neural_features_list, sentence_texts, block_indices = extract_session_data(mat_data)
        
        # BLOCK-WISE Z-SCORE NORMALIZATION (critical for handling drift)
        blockNums = np.array(block_indices)
        blockList = np.unique(blockNums)
        blocks = []
        for b in blockList:
            sentIdx = np.argwhere(blockNums == b).flatten()
            blocks.append(sentIdx)
        
        # Normalize each block separately
        for block_idx in blocks:
            # Concatenate all features in this block
            feats = np.concatenate([neural_features_list[i] for i in block_idx], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            
            # Normalize each trial in the block
            for i in block_idx:
                neural_features_list[i] = (neural_features_list[i] - feats_mean) / (feats_std + 1e-8)
        
        # Convert sentences to phoneme sequences using g2p
        g2p = G2p()
        phoneme_seqs = []
        phoneme_lens = []
        
        for sentence in sentence_texts:
            # Clean the transcription (same as paper)
            thisTranscription = sentence.strip()
            thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
            thisTranscription = thisTranscription.replace('--', '').lower()
            
            phonemes = []
            if len(thisTranscription) == 0:
                phonemes = ['SIL']
            else:
                for p in g2p(thisTranscription):
                    if p == ' ':
                        phonemes.append('SIL')
                    p = re.sub(r'[0-9]', '', p)  # Remove stress markers
                    if re.match(r'[A-Z]+', p):  # Only keep phonemes
                        phonemes.append(p)
                
                # Add SIL at the end (as per paper)
                phonemes.append('SIL')
            
            # Convert phonemes to IDs (1-40, 0 reserved for CTC blank)
            seqLen = len(phonemes)
            seqClassIDs = np.array([phoneToId(p) + 1 for p in phonemes], dtype=np.int32)
            
            phoneme_seqs.append(seqClassIDs)
            phoneme_lens.append(seqLen)
        
        return {
            'sentenceDat': neural_features_list,
            'phonemes': phoneme_seqs,
            'phoneLens': phoneme_lens,
            'sentenceText': sentence_texts,
            'blockIdx': block_indices
        }
            
    except Exception as e:
        print(f"Warning: Failed to process {mat_file}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'sentenceDat': [],
            'phonemes': [],
            'phoneLens': [],
            'sentenceText': [],
            'blockIdx': []
        }


def preprocess_competition_data(
    tar_gz_path: str,
    output_dir: str,
    n_input_features: int = 256
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract tar.gz to temporary directory
    print("Extracting competitionData.tar.gz...")
    extract_dir = output_dir / "temp_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # Find the competitionData directory
    comp_data_dir = extract_dir / "competitionData"
    if not comp_data_dir.exists():
        # Try to find it recursively
        comp_data_dir = list(extract_dir.rglob("competitionData"))[0]
    
    print(f"Found competition data at: {comp_data_dir}")
    
    # Process train, test, and holdout splits
    processed_data = {}
    
    for split in ['train', 'test', 'competitionHoldOut']:
        split_dir = comp_data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get all .mat files
        mat_files = sorted(list(split_dir.glob("*.mat")))
        
        if len(mat_files) == 0:
            print(f"Warning: No .mat files found in {split_dir}")
            continue
        
        # Group files by day (session)
        # Files are named like: t12.2022.04.28.mat
        day_groups = {}
        for mat_file in mat_files:
            day_key = mat_file.stem  # e.g., "t12.2022.04.28"
            if day_key not in day_groups:
                day_groups[day_key] = []
            day_groups[day_key].append(str(mat_file))
        
        # Process each day
        split_data = []
        for day_idx, (day_key, day_files) in enumerate(sorted(day_groups.items())):
            print(f"  Processing day {day_idx + 1}/{len(day_groups)}: {day_key}")
            day_data = process_day_data(day_files)
            
            # Ensure neural features are the correct size
            for i, feats in enumerate(day_data['sentenceDat']):
                if feats.shape[1] != n_input_features:
                    # Truncate or pad to n_input_features
                    if feats.shape[1] > n_input_features:
                        day_data['sentenceDat'][i] = feats[:, :n_input_features]
                    else:
                        pad_width = ((0, 0), (0, n_input_features - feats.shape[1]))
                        day_data['sentenceDat'][i] = np.pad(feats, pad_width, mode='constant')
            
            split_data.append(day_data)
        
        # Use standard names for splits
        if split == 'competitionHoldOut':
            processed_data['holdout'] = split_data
        else:
            processed_data[split] = split_data
    
    # Save preprocessed data
    output_file = output_dir / "competition_data.pkl"
    print(f"\nSaving preprocessed data to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nPreprocessing complete!")
    print(f"Train days: {len(processed_data.get('train', []))}")
    print(f"Test days: {len(processed_data.get('test', []))}")
    print(f"Holdout days: {len(processed_data.get('holdout', []))}")
    
    # Clean up temporary extraction
    import shutil
    shutil.rmtree(extract_dir)
    print("Cleaned up temporary files.")


def get_dataset_statistics(preprocessed_pkl: str) -> None:

    with open(preprocessed_pkl, 'rb') as f:
        data = pickle.load(f)
    
    print("\n=== Dataset Statistics ===")
    
    for split_name in ['train', 'test', 'holdout']:
        if split_name not in data:
            continue
        
        split_data = data[split_name]
        n_days = len(split_data)
        n_trials = sum(len(day['sentenceDat']) for day in split_data)
        
        # Get lengths
        neural_lens = [trial.shape[0] for day in split_data for trial in day['sentenceDat']]
        phoneme_lens = [plen for day in split_data for plen in day['phoneLens']]
        
        print(f"\n{split_name.upper()}:")
        print(f"  Days: {n_days}")
        print(f"  Trials: {n_trials}")
        print(f"  Neural sequence length: {np.mean(neural_lens):.1f} ± {np.std(neural_lens):.1f}")
        print(f"  Phoneme sequence length: {np.mean(phoneme_lens):.1f} ± {np.std(phoneme_lens):.1f}")


def unscrambleChans(timeSeriesDat: np.ndarray) -> np.ndarray:

    chanToElec = [
        63, 64, 62, 61, 59, 58, 60, 54, 57, 50, 53, 49, 52, 45, 55, 44,
        56, 39, 51, 43, 46, 38, 48, 37, 47, 36, 42, 35, 41, 34, 40, 33,
        96, 90, 95, 89, 94, 88, 93, 87, 92, 82, 86, 81, 91, 77, 85, 83,
        84, 78, 80, 73, 79, 74, 75, 76, 71, 72, 68, 69, 66, 70, 65, 67,
        128, 120, 127, 119, 126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113,
        112, 111, 109, 110, 107, 108, 106, 105, 104, 103, 102, 101, 100, 99, 97, 98,
        32, 30, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16,
        17, 7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 8
    ]
    chanToElec = np.array(chanToElec).astype(np.int32) - 1  # Convert to 0-indexing
    
    unscrambledDat = timeSeriesDat.copy()
    for x in range(len(chanToElec)):
        unscrambledDat[:, chanToElec[x]] = timeSeriesDat[:, x]
    
    return unscrambledDat


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess competitionData.tar.gz")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to competitionData.tar.gz")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for preprocessed data")
    parser.add_argument("--n_features", type=int, default=256,
                        help="Number of neural input features")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        pkl_file = Path(args.output) / "competition_data.pkl"
        get_dataset_statistics(str(pkl_file))
    else:
        preprocess_competition_data(args.input, args.output, args.n_features)

"""
Preprocess the already-extracted competition data.
"""

import pickle
from pathlib import Path
from src.dataio.preprocessing import process_day_data
import numpy as np

def preprocess_extracted_data(data_dir: str, output_dir: str, n_input_features: int = 256):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data = {}
    
    # Process each split
    for split in ['train', 'test', 'competitionHoldOut']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found at {split_dir}, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get all .mat files
        mat_files = sorted(list(split_dir.glob("*.mat")))
        
        if len(mat_files) == 0:
            print(f"Warning: No .mat files found in {split_dir}")
            continue
        
        print(f"  Found {len(mat_files)} .mat files")
        
        # Group files by day (each file is a day/session)
        split_data = []
        for idx, mat_file in enumerate(mat_files):
            day_key = mat_file.stem  # e.g., "t12.2022.04.28"
            print(f"  Processing {idx + 1}/{len(mat_files)}: {day_key}")
            
            # Process this day's data
            day_data = process_day_data([str(mat_file)])
            
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
        
        print(f"  Processed {len(split_data)} days for {split}")
    
    # Save preprocessed data
    output_file = output_dir / "competition_data.pkl"
    print(f"\nSaving preprocessed data to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nPreprocessing complete!")
    print(f"Output saved to: {output_file}")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for split_name, split_data in processed_data.items():
        total_sentences = sum(len(day['sentenceDat']) for day in split_data)
        print(f"{split_name}: {len(split_data)} days, {total_sentences} sentences")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess extracted competition data")
    parser.add_argument("--data_dir", type=str, default="data/competitionData",
                        help="Path to extracted competitionData directory")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for preprocessed data")
    parser.add_argument("--n_features", type=int, default=256,
                        help="Number of neural input features")
    
    args = parser.parse_args()
    
    preprocess_extracted_data(args.data_dir, args.output_dir, args.n_features)

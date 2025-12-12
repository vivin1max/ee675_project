import os
import time
import numpy as np
from src.classical.kalman_decoder import KalmanDecoderBaseline
from src.evaluation.metrics import compute_cer, compute_wer
from src.dataio.dataset import load_dataset
import yaml
import json


print('Loading config...')
CONFIG_PATH = os.environ.get('KALMAN_CONFIG', 'configs/kalman_baseline.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
print('Config loaded:', config)


print('Loading data...')
preprocessed_path = config['data']['preprocessed_path']
train_loader, test_loader, loaded_data = load_dataset(preprocessed_path, batch_size=1, num_workers=0, pin_memory=False)
print('Extracting features and labels...')
def extract_features_and_labels(split_data):
    neural = []
    labels = []
    for day in split_data:
        for x, y in zip(day['sentenceDat'], day['phonemes']):
            neural.append(np.array(x))
            labels.append(np.array(y))
    return neural, labels
train_neural, train_label = extract_features_and_labels(loaded_data['train'])
test_neural, test_label = extract_features_and_labels(loaded_data['test'])
print(f'  Train sentences: {len(train_neural)}')
print(f'  Test sentences: {len(test_neural)}')
print(f'  Example train feature shape: {train_neural[0].shape}')
print(f'  Example train label shape: {train_label[0].shape}')
# Get symbol set from all train and test labels
all_labels = []
for split in ['train', 'test']:
    for day in loaded_data[split]:
        for y in day['phonemes']:
            all_labels.extend(y.tolist())
max_label = max(all_labels)
symbol_set = list(range(max_label + 1))
print(f'  Symbol set size: {len(symbol_set)}')


print('Initializing Kalman decoder...')
latent_dim = config['model'].get('latent_dim', 16)
use_pca = config['model'].get('use_pca', True)
ridge_alpha = config['model'].get('ridge_alpha', 0.0)
decoder = KalmanDecoderBaseline(latent_dim=latent_dim, use_pca=use_pca, ridge_alpha=ridge_alpha)

print('Fitting Kalman decoder...')
start_time = time.time()
decoder.fit(train_neural, train_label, symbol_set)
training_time = (time.time() - start_time) / 3600.0  # hours
print(f'  Training complete. Time: {training_time:.2f} hours')


print('Decoding test set...')
pred_seqs = []
for i, x in enumerate(test_neural):
    if i % 10 == 0:
        print(f'  Decoding sentence {i+1}/{len(test_neural)} | shape: {x.shape}')
    try:
        pred = decoder.predict([x])[0]
    except Exception as e:
        print(f'    Error decoding sentence {i}: {e}')
        pred = np.zeros(x.shape[0], dtype=int)
    pred_seqs.append(pred)
print('  Decoding complete.')

print('Computing metrics...')
cer = compute_cer(pred_seqs, test_label)
wer = compute_wer(pred_seqs, test_label)
print(f'  CER: {cer}')
print(f'  WER: {wer}')

# Logging
output_dir = config['logging'].get('output_dir', 'results/kalman_baseline')
os.makedirs(output_dir, exist_ok=True)
summary = {
    'model_name': 'kalman_baseline',
    'CER': cer,
    'WER': wer,
    'training_time': training_time
}
with open(os.path.join(output_dir, 'kalman_baseline_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('Kalman Baseline Results:')
print(json.dumps(summary, indent=2))

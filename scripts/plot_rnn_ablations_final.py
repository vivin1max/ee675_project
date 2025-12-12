"""
Generate side-by-side RNN ablation study plots (CER and WER comparison).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('results/final_results.csv')

# Filter for RNN ablation models only
rnn_models = [
    'rnn_baseline',
    'rnn_fastemit',
    'rnn_coord_dropout',
    'rnn_layernorm',
    'rnn_step_lr',
    'rnn_best_combo'
]

df_rnn = df[df['experiment_name'].isin(rnn_models)].copy()

# Rename models for clean plotting
name_map = {
    'rnn_baseline': 'RNN Baseline',
    'rnn_fastemit': '+FastEmit',
    'rnn_coord_dropout': '+CoordDrop',
    'rnn_layernorm': '+LayerNorm',
    'rnn_step_lr': '+StepLR',
    'rnn_best_combo': '+BestCombo'
}

df_rnn['display_name'] = df_rnn['experiment_name'].map(name_map)

# Convert to percentages
df_rnn['CER'] = df_rnn['final_CER'] * 100
df_rnn['WER'] = df_rnn['final_WER'] * 100

# Sort by CER and WER for each subplot
df_cer_sorted = df_rnn.sort_values('CER')
df_wer_sorted = df_rnn.sort_values('WER')

# Print performance tables
print('='*70)
print('RNN ABLATION STUDY — PERFORMANCE SUMMARY')
print('='*70)
print('\nModels sorted by CER (best → worst):')
print('-'*70)
for idx, row in df_cer_sorted.iterrows():
    print(f'{row["display_name"]:20s} | CER: {row["CER"]:5.2f}%')

print('\n' + '='*70)
print('Models sorted by WER (best → worst):')
print('-'*70)
for idx, row in df_wer_sorted.iterrows():
    print(f'{row["display_name"]:20s} | WER: {row["WER"]:5.2f}%')
print('='*70 + '\n')

# Create figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Use Blues color palette
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df_rnn)))

# LEFT SUBPLOT: CER Comparison
bars1 = ax1.barh(df_cer_sorted['display_name'], df_cer_sorted['CER'], color=colors)
ax1.set_xlabel('Character Error Rate (%)', fontsize=12)
ax1.set_title('RNN Ablation Study — CER Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
ax1.invert_yaxis()  # Best on top

# Annotate bars with CER values
for i, (bar, val) in enumerate(zip(bars1, df_cer_sorted['CER'])):
    ax1.text(val + 0.5, i, f'{val:.2f}', va='center', fontsize=10)

# RIGHT SUBPLOT: WER Comparison
bars2 = ax2.barh(df_wer_sorted['display_name'], df_wer_sorted['WER'], color=colors)
ax2.set_xlabel('Word Error Rate (%)', fontsize=12)
ax2.set_title('RNN Ablation Study — WER Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
ax2.invert_yaxis()  # Best on top

# Annotate bars with WER values
for i, (bar, val) in enumerate(zip(bars2, df_wer_sorted['WER'])):
    ax2.text(val + 0.5, i, f'{val:.2f}', va='center', fontsize=10)

# Rotate y-labels if needed 
plt.setp(ax1.get_yticklabels(), fontsize=11)
plt.setp(ax2.get_yticklabels(), fontsize=11)

plt.tight_layout()

# Save figure
output_path = 'plots/rnn_ablations_side_by_side.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Plot saved to: {output_path}')
print(f'Figure size: 14x6 inches, 300 DPI')
plt.close()

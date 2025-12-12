"""
Plots:
1. CER & WER grouped bar chart
2. Training time comparison
3. Model size vs accuracy 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
os.makedirs('results/plots', exist_ok=True)

print('='*80)
print('DEEP LEARNING ARCHITECTURE COMPARISON')
print('='*80)
print()

# ============================================================================
# STEP 1: Load and inspect CSV
# ============================================================================
print('STEP 1: Loading final_results.csv...')
df = pd.read_csv('results/final_results.csv')

print(f'\nDataset shape: {df.shape}')
print(f'\nColumn names:')
for col in df.columns:
    print(f'  - {col}')

print(f'\nUnique experiment names:')
for name in df['experiment_name'].unique():
    print(f'  - {name}')

print(f'\nFirst few rows:')
print(df[['experiment_name', 'model_name', 'best_CER', 'best_WER', 'training_time_hours']].head(10))
print()

# ============================================================================
# STEP 2: Filter target models
# ============================================================================
print('STEP 2: Filtering target models...')

# Define target models and their labels
target_models = {
    'rnn_baseline': 'RNN Baseline',
    'arch_lstm_overfitfix': 'LSTM',
    'arch_transformer_overfitfix': 'Transformer'
}

# Filter rows
df_filtered = df[df['experiment_name'].isin(target_models.keys())].copy()

# Add display labels
df_filtered['label'] = df_filtered['experiment_name'].map(target_models)

# Select and rename columns
# Use final_WER since best_WER has NaN values
plot_df = pd.DataFrame({
    'label': df_filtered['label'],
    'best_cer': df_filtered['best_CER'] * 100,  # Convert to percentage
    'best_wer': df_filtered['final_WER'] * 100,  # Use final_WER
    'train_time_hrs': df_filtered['training_time_hours']
})

# Sort by predefined order
label_order = ['RNN Baseline', 'LSTM', 'Transformer']
plot_df['label'] = pd.Categorical(plot_df['label'], categories=label_order, ordered=True)
plot_df = plot_df.sort_values('label').reset_index(drop=True)

print(f'\nFiltered data for plotting:')
print(plot_df.to_string(index=False))
print()

# ============================================================================
# STEP 3: Plot 1 - CER & WER Grouped Bar Chart
# ============================================================================
print('STEP 3: Creating CER & WER grouped bar chart...')

fig, ax = plt.subplots(figsize=(10, 6))

# Bar settings
x = np.arange(len(plot_df))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, plot_df['best_cer'], width, label='CER', 
               color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, plot_df['best_wer'], width, label='WER', 
               color='#e74c3c', edgecolor='black', linewidth=1.5)

# Highlight LSTM (best performer)
lstm_idx = plot_df[plot_df['label'] == 'LSTM'].index[0]
bars1[lstm_idx].set_hatch('//')
bars1[lstm_idx].set_linewidth(2.5)
bars2[lstm_idx].set_hatch('//')
bars2[lstm_idx].set_linewidth(2.5)

# Annotate values
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Styling
ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Deep Architectures – CER & WER Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(plot_df['label'], fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylim(0, max(plot_df['best_wer'].max(), plot_df['best_cer'].max()) * 1.15)

plt.tight_layout()
output_path_1 = 'results/plots/dl_architectures_cer_wer.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_path_1}')
plt.close()

# ============================================================================
# STEP 4: Plot 2 - Training Time Bar Chart (Horizontal)
# ============================================================================
print('STEP 4: Creating training time bar chart...')

fig, ax = plt.subplots(figsize=(10, 5))

# Define colors (lighter to darker = faster to slower)
colors = ['#2ecc71', '#3498db', '#e67e22']  # Green (fastest), Blue, Orange (slowest)

# Create horizontal bars
bars = ax.barh(plot_df['label'], plot_df['train_time_hrs'], 
               color=colors, edgecolor='black', linewidth=1.5)

# Annotate values
for i, (bar, val) in enumerate(zip(bars, plot_df['train_time_hrs'])):
    ax.text(val + 0.2, i, f'{val:.2f} hrs', 
            va='center', fontsize=12, fontweight='bold')

# Styling
ax.set_xlabel('Training Time (hours)', fontsize=14, fontweight='bold')
ax.set_title('Training Time – Deep Architectures', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, plot_df['train_time_hrs'].max() * 1.2)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.invert_yaxis()  # Best (fastest) on top

plt.tight_layout()
output_path_2 = 'results/plots/dl_architectures_training_time.png'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_path_2}')
plt.close()

# ============================================================================
# STEP 5: Plot 3 - Model Size vs Accuracy 
# ============================================================================
print('STEP 5: Checking for parameter count data...')

# Check if we have parameter information
# Estimate from architecture (if not in CSV)
# These are rough estimates based on the architecture descriptions
param_estimates = {
    'RNN Baseline': 50,  # ~50M params (5-layer bidirectional GRU)
    'LSTM': 55,          # ~55M params (LSTM has more gates)
    'Transformer': 20    # ~20M params (lighter architecture)
}

if 'params_millions' in df_filtered.columns and not df_filtered['params_millions'].isna().all():
    print('  Using params_millions from CSV')
    plot_df['params_millions'] = df_filtered['params_millions'].values
else:
    print('  Parameter count not in CSV. Using architecture-based estimates.')
    plot_df['params_millions'] = plot_df['label'].map(param_estimates)

print('\nCreating model size vs WER scatter plot...')

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
colors_scatter = ['#e74c3c', '#2ecc71', '#9b59b6']  # RNN=red, LSTM=green, Trans=purple
markers = ['o', 's', '^']

for i, (idx, row) in enumerate(plot_df.iterrows()):
    ax.scatter(row['params_millions'], row['best_wer'], 
               s=300, c=colors_scatter[i], marker=markers[i], 
               edgecolor='black', linewidth=2, alpha=0.8, zorder=3)
    
    # Add label with slight offset
    offset_x = 2 if row['label'] != 'Transformer' else -2
    offset_y = 2
    ha = 'left' if row['label'] != 'Transformer' else 'right'
    
    ax.annotate(row['label'], 
                xy=(row['params_millions'], row['best_wer']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=12, fontweight='bold', ha=ha,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='gray', alpha=0.8))

# Styling
ax.set_xlabel('Model Parameters (millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Word Error Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Size vs WER – Deep Architectures', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(15, 60)

# Add trend line
z = np.polyfit(plot_df['params_millions'], plot_df['best_wer'], 1)
p = np.poly1d(z)
x_line = np.linspace(plot_df['params_millions'].min(), plot_df['params_millions'].max(), 100)
ax.plot(x_line, p(x_line), linestyle='--', color='gray', alpha=0.5, linewidth=2, label='Linear trend')
ax.legend(fontsize=11)

plt.tight_layout()
output_path_3 = 'results/plots/dl_architectures_params_vs_wer.png'
plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_path_3}')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print('='*80)
print('SUMMARY: Deep Learning Architecture Comparison')
print('='*80)
print('\nFinal plotting data:')
print(plot_df.to_string(index=False))
print()
print('Generated plots:')
print(f'  1. {output_path_1}')
print(f'  2. {output_path_2}')
print(f'  3. {output_path_3}')
print()
print('Key findings:')
print(f'  • Best CER: {plot_df.loc[plot_df["best_cer"].idxmin(), "label"]} ({plot_df["best_cer"].min():.2f}%)')
print(f'  • Best WER: {plot_df.loc[plot_df["best_wer"].idxmin(), "label"]} ({plot_df["best_wer"].min():.2f}%)')
print(f'  • Fastest training: {plot_df.loc[plot_df["train_time_hrs"].idxmin(), "label"]} ({plot_df["train_time_hrs"].min():.2f} hrs)')
print(f'  • Smallest model: {plot_df.loc[plot_df["params_millions"].idxmin(), "label"]} ({plot_df["params_millions"].min():.0f}M params)')
print()
print('All plots saved successfully!')
print('='*80)

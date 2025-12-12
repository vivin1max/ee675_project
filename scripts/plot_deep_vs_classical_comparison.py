"""
Plots:
1. CER & WER grouped bar chart (all 5 models)
2. Model size vs WER scatter plot (all 5 models)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
os.makedirs('results/plots', exist_ok=True)

print('='*80)
print('DEEP LEARNING VS CLASSICAL MODELS COMPARISON')
print('='*80)
print()

# ============================================================================
# STEP 1: Load deep learning results from CSV
# ============================================================================
print('STEP 1: Loading deep learning results from final_results.csv...')
df_full = pd.read_csv('results/final_results.csv')

# Filter for the three deep learning models
deep_models = ['rnn_baseline', 'arch_lstm_overfitfix', 'arch_transformer_overfitfix']
df_deep = df_full[df_full['experiment_name'].isin(deep_models)].copy()

# Extract relevant columns (use final_WER since best_WER has NaN)
deep_data = []
for _, row in df_deep.iterrows():
    deep_data.append({
        'model_name': row['experiment_name'],
        'best_cer': row['best_CER'],
        'final_wer': row['final_WER'],
        'params_million': 50 if 'baseline' in row['experiment_name'] else (55 if 'lstm' in row['experiment_name'] else 20),
        'family': 'Deep'
    })

print(f'  Loaded {len(deep_data)} deep learning models')

# ============================================================================
# STEP 2: Hardcode classical model metrics
# ============================================================================
print('STEP 2: Adding hardcoded classical model metrics...')

classical_data = [
    {
        'model_name': 'kalman_baseline',
        'best_cer': 0.81,
        'final_wer': 0.81,
        'params_million': 0.07,  # ~70K params
        'family': 'Classical'
    },
    {
        'model_name': 'kalman_ridge_test',
        'best_cer': 0.7729,
        'final_wer': 0.7729,
        'params_million': 0.077,  # ~77K params
        'family': 'Classical'
    }
]

print(f'  Added {len(classical_data)} classical models')

# ============================================================================
# STEP 3: Combine into single DataFrame
# ============================================================================
print('STEP 3: Building combined DataFrame...')

# Combine data
all_data = deep_data + classical_data
df = pd.DataFrame(all_data)

# Add display names
label_map = {
    'rnn_baseline': 'RNN Baseline',
    'arch_lstm_overfitfix': 'LSTM',
    'arch_transformer_overfitfix': 'Transformer',
    'kalman_baseline': 'Kalman Baseline',
    'kalman_ridge_test': 'Kalman Ridge'
}
df['display_name'] = df['model_name'].map(label_map)

# Convert to percentages
df['cer_pct'] = df['best_cer'] * 100
df['wer_pct'] = df['final_wer'] * 100

# Set display order
order = ['RNN Baseline', 'LSTM', 'Transformer', 'Kalman Baseline', 'Kalman Ridge']
df['display_name'] = pd.Categorical(df['display_name'], categories=order, ordered=True)
df = df.sort_values('display_name').reset_index(drop=True)

print('\nCombined DataFrame:')
print(df[['display_name', 'cer_pct', 'wer_pct', 'params_million', 'family']].to_string(index=False))
print()

# ============================================================================
# STEP 4: Plot 1 - CER & WER Grouped Bar Chart
# ============================================================================
print('STEP 4: Creating CER & WER grouped bar chart...')

fig, ax = plt.subplots(figsize=(12, 6))

# Bar settings
x = np.arange(len(df))
width = 0.35

# Colors: CER = blue, WER = red (consistent with prior plots)
cer_color = '#3498db'
wer_color = '#e74c3c'

# Create bars
bars1 = ax.bar(x - width/2, df['cer_pct'], width, label='CER', 
               color=cer_color, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, df['wer_pct'], width, label='WER', 
               color=wer_color, edgecolor='black', linewidth=1.5)

# Add a visual separator between Deep and Classical
# Draw vertical line after Transformer (index 2)
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
ax.text(1.25, max(df['wer_pct'].max(), df['cer_pct'].max()) * 0.95, 
        'Deep Learning', ha='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
ax.text(3.5, max(df['wer_pct'].max(), df['cer_pct'].max()) * 0.95, 
        'Classical', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.3))

# Annotate values on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.0,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.0,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Styling
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Deep Learning vs Classical – CER & WER Comparison', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['display_name'], fontsize=11, rotation=25, ha='right')
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylim(0, max(df['wer_pct'].max(), df['cer_pct'].max()) * 1.15)

plt.tight_layout()
output_path_1 = 'results/plots/deep_vs_classical_cer_wer.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_path_1}')
plt.close()

# ============================================================================
# STEP 5: Plot 2 - Model Size vs WER Scatter Plot
# ============================================================================
print('STEP 5: Creating model size vs WER scatter plot...')

fig, ax = plt.subplots(figsize=(12, 7))

# Separate deep and classical for different styling
df_deep_plot = df[df['family'] == 'Deep']
df_classical_plot = df[df['family'] == 'Classical']

# Plot deep learning models
ax.scatter(df_deep_plot['params_million'], df_deep_plot['wer_pct'], 
           s=400, c='#3498db', marker='o', 
           edgecolor='black', linewidth=2, alpha=0.7, 
           label='Deep Learning', zorder=3)

# Plot classical models
ax.scatter(df_classical_plot['params_million'], df_classical_plot['wer_pct'], 
           s=400, c='#f39c12', marker='s', 
           edgecolor='black', linewidth=2, alpha=0.7, 
           label='Classical', zorder=3)

# Annotate each point with model name
for _, row in df.iterrows():
    # Offset annotation to avoid overlap
    if row['family'] == 'Classical':
        offset_x = 2
        offset_y = -2
        ha = 'left'
    else:
        offset_x = 0
        offset_y = 3
        ha = 'center'
    
    ax.annotate(row['display_name'], 
                xy=(row['params_million'], row['wer_pct']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=11, fontweight='bold', ha=ha,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='gray', alpha=0.8))

# Add linear trend line across all points
z = np.polyfit(df['params_million'], df['wer_pct'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['params_million'].min() - 1, df['params_million'].max() + 5, 100)
ax.plot(x_line, p(x_line), linestyle='--', color='gray', alpha=0.5, 
        linewidth=2, label=f'Linear trend: y={z[0]:.2f}x+{z[1]:.1f}')

# Styling
ax.set_xlabel('Model Parameters (millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Word Error Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Size vs WER – Deep vs Classical', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

# Set x-axis to log scale for better visualization
ax.set_xscale('log')
ax.set_xlim(0.05, 70)

plt.tight_layout()
output_path_2 = 'results/plots/deep_vs_classical_size_vs_wer.png'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f'✓ Saved: {output_path_2}')
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print('='*80)
print('SUMMARY: Deep Learning vs Classical Comparison')
print('='*80)
print('\nModel Performance:')
print(df[['display_name', 'cer_pct', 'wer_pct', 'params_million', 'family']].to_string(index=False))
print()
print('Generated plots:')
print(f'  1. {output_path_1}')
print(f'  2. {output_path_2}')
print()
print('Key findings:')
print(f'  • Best WER (Deep):      {df[df["family"]=="Deep"]["display_name"].iloc[df[df["family"]=="Deep"]["wer_pct"].idxmin()]} ({df[df["family"]=="Deep"]["wer_pct"].min():.2f}%)')
print(f'  • Best WER (Classical): {df[df["family"]=="Classical"]["display_name"].iloc[df[df["family"]=="Classical"]["wer_pct"].idxmin() - 3]} ({df[df["family"]=="Classical"]["wer_pct"].min():.2f}%)')
print(f'  • Parameter ratio:      Deep models are {df[df["family"]=="Deep"]["params_million"].mean() / df[df["family"]=="Classical"]["params_million"].mean():.0f}x larger')
print(f'  • Performance gap:      Deep models achieve {(df[df["family"]=="Classical"]["wer_pct"].min() - df[df["family"]=="Deep"]["wer_pct"].min()):.1f}% lower WER')
print()
print('All plots saved successfully!')
print('='*80)

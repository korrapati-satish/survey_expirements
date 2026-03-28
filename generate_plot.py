import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the experiment results
data = {
    'Model': ['Stackelberg', 'Evolutionary', 'Baseline (Nash)', 'Baseline (Static)'],
    'Defender Payoff': [13.362563, 13.053609, 13.345522, 12.699532],
    'ADR (%)': [96.943611, 95.491355, 97.174858, 95.453555],
    'FPR (%)': [5.443110, 5.082197, 5.443110, 2.867464]
}
df = pd.DataFrame(data)

# Create the plot with 3 subplots
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Plot 1: Defender Payoff
sns.barplot(x='Model', y='Defender Payoff', data=df, ax=axes[0], palette='viridis', hue='Model', dodge=False)
axes[0].set_title('Defender Payoff Comparison Across Game-Theoretic Models', fontsize=16)
axes[0].set_xlabel('')
axes[0].set_ylabel('Average Defender Payoff (Utility)', fontsize=12)
axes[0].set_ylim(12, 13.6)
axes[0].legend().set_visible(False)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.4f')

# Plot 2: ADR
sns.barplot(x='Model', y='ADR (%)', data=df, ax=axes[1], palette='plasma', hue='Model', dodge=False)
axes[1].set_title('Attack Detection Rate (ADR) Comparison', fontsize=16)
axes[1].set_xlabel('')
axes[1].set_ylabel('ADR (%)', fontsize=12)
axes[1].set_ylim(94, 98)
axes[1].legend().set_visible(False)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.2f%%')

# Plot 3: FPR
sns.barplot(x='Model', y='FPR (%)', data=df, ax=axes[2], palette='magma', hue='Model', dodge=False)
axes[2].set_title('False Positive Rate (FPR) Comparison', fontsize=16)
axes[2].set_xlabel('Game-Theoretic Model', fontsize=12)
axes[2].set_ylabel('FPR (%)', fontsize=12)
axes[2].set_ylim(0, 6)
axes[2].legend().set_visible(False)
for container in axes[2].containers:
    axes[2].bar_label(container, fmt='%.2f%%')


plt.tight_layout()
plt.savefig('model_comparison_charts.pdf')

print("Plots saved to model_comparison_charts.pdf")

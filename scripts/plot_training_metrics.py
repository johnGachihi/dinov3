import json
import matplotlib.pyplot as plt

# Define your runs
log_files = {
  'Current DinoV3 on Sen2Venus': '/home/admin/john/SR4Seg/dinov3/logs/fixed_norm/training_metrics.json',
  'Previous': '/home/admin/john/SR4Seg/dinov3/logs/sr/training_metrics.json',
  # 'Higher LR': 'metrics_high_lr.jsonl',
}

# Metrics to compare
metrics_to_plot = ['total_loss', 'ibot_loss', 'dino_local_crops_loss', 'dino_global_crops_loss']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
  ax = axes[idx]

  for run_name, filepath in log_files.items():
    # Load metrics
    with open(filepath, 'r') as f:
      data = [json.loads(line) for line in f]

    iterations = [m['iteration'] for m in data]
    values = [m[metric] for m in data]

    ax.plot(iterations, values, label=run_name, linewidth=2, alpha=0.8)

  ax.set_xlabel('Iteration', fontsize=11)
  ax.set_ylabel('Value', fontsize=11)
  ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
  ax.legend()
  ax.grid(True, alpha=0.3)

  if 'grad_norm' in metric:
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('multi_run_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
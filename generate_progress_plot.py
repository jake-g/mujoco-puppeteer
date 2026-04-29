import os
import re

import matplotlib.pyplot as plt


def generate_plot():
  results_dir = "results"
  import glob
  files = glob.glob(os.path.join(results_dir, "evolution_history_*.tsv"))
  if not files:
    print("No history files found.")
    return

  data = {}
  for f_path in files:
    with open(f_path, "r") as f:
      # Skip header
      f.readline()
      for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
          species = parts[1]
          gen = int(parts[2])
          reward = float(parts[3])

          if species not in data:
            data[species] = {}
          data[species][gen] = reward

  # Plot data.
  plt.figure(figsize=(12, 8))

  for species, points in data.items():
    if not points:
      continue
    # Sort by generation
    sorted_gens = sorted(points.keys())
    rewards = [points[g] for g in sorted_gens]

    plt.plot(sorted_gens, rewards, label=species, marker='o', markersize=4)

  # Add markers for notable events (Curriculum)
  plt.axvline(x=5,
              color='gray',
              linestyle='--',
              alpha=0.7,
              label='Gen 5: Rough Terrain')
  plt.axvline(x=10,
              color='blue',
              linestyle='--',
              alpha=0.7,
              label='Gen 10: Wind Enabled')

  plt.xlabel("Generation")
  plt.ylabel("Best Reward (Fitness)")
  plt.title("Hill Climbing: Species Improvement Over Time")
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()

  os.makedirs("results", exist_ok=True)
  plot_path = "results/progress.png"
  plt.savefig(plot_path, dpi=300)
  print(f"Success! Progress plot saved to {plot_path}")


if __name__ == "__main__":
  generate_plot()

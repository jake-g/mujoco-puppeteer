import hashlib
import os

import yaml


def clean_duplicates(results_dir="results"):
  if not os.path.exists(results_dir):
    print(f"Results directory {results_dir} does not exist.")
    return

  folders = [
      os.path.join(results_dir, d)
      for d in os.listdir(results_dir)
      if os.path.isdir(os.path.join(results_dir, d))
  ]

  stats = {}

  for folder in folders:
    folder_name = os.path.basename(folder)
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".ppm")
    ]
    files.sort()

    hashes = {}
    duplicates = 0

    for file_path in files:
      try:
        with open(file_path, "rb") as f:
          file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
          # Duplicate. Delete it!
          os.remove(file_path)
          duplicates += 1
        else:
          hashes[file_hash] = file_path
      except Exception as e:
        print(f"Failed to process {file_path}: {e}")

    remaining = len(files) - duplicates
    stats[folder_name] = remaining
    print(
        f"Folder {folder_name}: Deleted {duplicates} duplicates. Remaining: {remaining}"
    )

  # Compute species counts
  species_counts = {}
  import re
  for folder_name, count in stats.items():
    if "__" in folder_name:
      species = folder_name.split("__")[0]
    else:
      # Handle hex IDs like _000a or _0001
      species = re.sub(r"_[0-9a-f]+$", "", folder_name)
      species = re.sub(r"_default$", "", species)

    if species not in species_counts:
      species_counts[species] = 0
    species_counts[species] += 1

  # Save stats to a file.
  stats_path = os.path.join(results_dir, "index.yaml")
  full_report = {
      "summary": {
          "total_folders": len(stats),
          "total_files": sum(stats.values())
      },
      "species_counts": species_counts,
      "folder_sizes": stats
  }

  with open(stats_path, "w") as f:
    yaml.dump(full_report, f, sort_keys=False)

  print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
  clean_duplicates()

import glob
import hashlib
import logging
import os
import re
import subprocess
import sys

import yaml

from auto_evolve import update_leaderboard
from render import create_gif
from render import generate_plot
from render import render_template
from render import rerender_all
from render import rerender_scenes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LOCK_FILE = "maintenance.lock"


def clean_duplicates(results_dir="results"):
  """Deduplicates PPM files in results and saves an index."""
  if not os.path.exists(results_dir):
    print(f"Results directory {results_dir} does not exist.")
    return

  folders = []
  agents_dirs = []

  # Live agents folder
  agents_dir = os.path.join(results_dir, "agents")
  if os.path.exists(agents_dir):
    agents_dirs.append(agents_dir)

  # Archived results_* folders
  for d in os.listdir(results_dir):
    if d.startswith("results_") and os.path.isdir(os.path.join(results_dir, d)):
      folders.append(os.path.join(results_dir, d))
      v_agents_dir = os.path.join(results_dir, d, "agents")
      if os.path.exists(v_agents_dir):
        agents_dirs.append(v_agents_dir)

  # Scan all identified agents directories
  for a_dir in agents_dirs:
    species_dirs = [
        os.path.join(a_dir, d)
        for d in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, d))
    ]
    for species_dir in species_dirs:
      generations_dir = os.path.join(species_dir, "generations")
      if os.path.exists(generations_dir):
        folders.extend([
            os.path.join(generations_dir, d)
            for d in os.listdir(generations_dir)
            if os.path.isdir(os.path.join(generations_dir, d))
        ])

  # Handle demo folders in results root
  folders.extend([
      os.path.join(results_dir, d)
      for d in os.listdir(results_dir)
      if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("demo")
  ])

  stats = {}

  for folder in folders:
    folder_name = os.path.basename(folder)
    files = []
    for root, _, fs in os.walk(folder):
      for f in fs:
        if f.endswith(".ppm") or f.endswith(".jpg") or f.endswith(".yaml"):
          files.append(os.path.join(root, f))
    files.sort()

    hashes = {}
    duplicates = 0

    for file_path in files:
      try:
        with open(file_path, "rb") as f:
          file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
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

  species_counts = {}
  for folder_name, count in stats.items():
    if "__" in folder_name:
      species = folder_name.split("__")[0]
    else:
      species = re.sub(r"_[0-9a-f]+$", "", folder_name)
      species = re.sub(r"_default$", "", species)

    if species not in species_counts:
      species_counts[species] = 0
    species_counts[species] += 1

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


def render_evolved():
  evolved_dir = "templates/agents_evolved"
  if not os.path.exists(evolved_dir):
    print("No evolved agents folder found.")
    return

  files = glob.glob(os.path.join(evolved_dir, "*.yaml"))
  print(f"Found {len(files)} evolved agents to render.")

  for f_path in files:
    jpg_path = f_path.replace(".yaml", ".jpg")
    print(f"Rendering {f_path}...")
    try:
      render_template(f_path, jpg_path, output_format="jpg")
    except Exception as e:
      print(f"Failed to render {f_path}: {e}")


def run_maintenance(run_tests=False):
  if os.path.exists(LOCK_FILE):
    logger.info("Another instance of maintenance.py is running. Exiting.")
    sys.exit(0)

  with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

  try:
    logger.info("=== Starting Scheduled Maintenance ===")

    if run_tests:
      logger.info("Running unit tests...")
      test_res = subprocess.run(["make", "test"],
                                capture_output=True,
                                text=True)
      if test_res.returncode != 0:
        logger.error("Tests failed!")
        logger.error(test_res.stderr)
      else:
        logger.info("All tests passed successfully.")

    # Clean Results
    logger.info("Cleaning and indexing results...")
    try:
      clean_duplicates()
    except Exception as e:
      logger.error("Failed to clean results: %s", e)

    # Update Leaderboard
    logger.info("Updating leaderboard...")
    try:
      update_leaderboard()
    except Exception as e:
      logger.error("Failed to update leaderboard: %s", e)

    # Create Evolution GIF
    logger.info("Creating evolution GIFs...")
    try:
      create_gif()
    except Exception as e:
      logger.error("Failed to create GIF: %s", e)

    # Generate Progress Plot
    logger.info("Generating progress plot...")
    try:
      generate_plot()
    except Exception as e:
      logger.error("Failed to generate plot: %s", e)

    # Render all templates and evolved agents
    logger.info("Rendering all templates and evolved agents...")
    try:
      rerender_all(use_gif=True)
      rerender_scenes()
      render_evolved()
    except Exception as e:
      logger.error("Failed to render agents: %s", e)

    logger.info("=== Maintenance Finished ===")
  finally:
    if os.path.exists(LOCK_FILE):
      os.remove(LOCK_FILE)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--tests",
                      action="store_true",
                      help="Run tests during maintenance")
  args = parser.parse_args()

  run_maintenance(run_tests=args.tests)

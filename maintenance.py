import logging
import os
import sys

from auto_evolve import update_leaderboard
from clean_results import clean_duplicates
from create_evolution_gif import create_gif
from generate_progress_plot import generate_plot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LOCK_FILE = "maintenance.lock"


def run_maintenance():
  if os.path.exists(LOCK_FILE):
    logger.info("Another instance of maintenance.py is running. Exiting.")
    sys.exit(0)

  with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

  try:
    logger.info("=== Starting Scheduled Maintenance ===")

    # 1. Clean Results (Deduplicate and Index)
    logger.info("Cleaning and indexing results...")
    try:
      clean_duplicates()
    except Exception as e:
      logger.error("Failed to clean results: %s", e)

    # 2. Create Evolution GIF
    logger.info("Creating evolution GIF...")
    try:
      create_gif()
    except Exception as e:
      logger.error("Failed to create GIF: %s", e)

    # 3. Generate Progress Plot
    logger.info("Generating progress plot...")
    try:
      generate_plot()
    except Exception as e:
      logger.error("Failed to generate plot: %s", e)

    # 4. Update Leaderboard
    logger.info("Updating leaderboard...")
    try:
      update_leaderboard()
    except Exception as e:
      logger.error("Failed to update leaderboard: %s", e)

    logger.info("=== Maintenance Finished ===")
  finally:
    if os.path.exists(LOCK_FILE):
      os.remove(LOCK_FILE)


if __name__ == "__main__":
  run_maintenance()

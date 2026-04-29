import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_cron():
  logger.info("=== Starting Scheduled Cron Maintenance ===")

  # 1. Run Tests
  logger.info("Running unit tests...")
  test_res = subprocess.run(["make", "test"], capture_output=True, text=True)
  if test_res.returncode != 0:
    logger.error("Tests failed during cron run!")
    logger.error(test_res.stderr)
  else:
    logger.info("All tests passed successfully.")

  # 2. Clean Results (Deduplicate and Index)
  logger.info("Cleaning and indexing results...")
  clean_res = subprocess.run(["make", "clean-results"],
                             capture_output=True,
                             text=True)
  logger.info(clean_res.stdout)

  # 3. Update Leaderboard
  logger.info("Updating leaderboard...")
  leaderboard_res = subprocess.run(["python3", "update_leaderboard.py"],
                                   capture_output=True,
                                   text=True)
  if leaderboard_res.returncode != 0:
    logger.error("Failed to update leaderboard!")
    logger.error(leaderboard_res.stderr)
  else:
    logger.info("Leaderboard updated successfully.")

  logger.info("=== Cron Maintenance Finished ===")


if __name__ == "__main__":
  run_cron()

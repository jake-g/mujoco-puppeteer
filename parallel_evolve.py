import os
import subprocess
import time

from auto_evolve import update_leaderboard
from maintenance import run_maintenance

species_list = [
    "aegis_turtle", "arachne_spider", "centipede", "chimera", "crab",
    "dragon", "ein_corgi", "elephant", "frog", "giraffe",
    "giraffe_default", "goliath_crawler", "gorilla", "hercules", "hexapod",
    "horse", "kangaroo", "khepri_beetle", "legion_hexapod", "mantis",
    "mech_biped", "megapede", "megarachne", "ostrich", "quadruped",
    "scorpion", "scorpion_king", "snake", "starfish", "stilts_biped",
    "stingray", "tarantula", "urchin", "asymmetric_quadruped", "rolling_agent",
    "quadruped_fixed"
]

max_parallel = 6
active_processes: list[subprocess.Popen] = []
print(f"Starting parallel evolution manager. Max parallel: {max_parallel}")

for species in species_list:
  while len(active_processes) >= max_parallel:
    # Wait for a slot to open
    for p in active_processes:
      if p.poll() is not None:
        active_processes.remove(p)
        print(f"Process finished.")
    time.sleep(1)

  print(f"Starting evolution for {species}...")
  p = subprocess.Popen([
      ".venv/bin/python3", "auto_evolve.py", "--species", species, "--pop-size",
      "50", "--generations", "100"
  ])
  active_processes.append(p)

# Wait for all to finish
while active_processes:
  for p in active_processes:
    if p.poll() is not None:
      active_processes.remove(p)
  time.sleep(1)

print("All species evolved!")
run_maintenance(run_tests=True)

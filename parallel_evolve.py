import os
import subprocess
import time

from auto_evolve import update_leaderboard
from maintenance import run_maintenance

species_list = [
    "chimera", "tarantula", "stingray", "mech_biped", "crab", "megapede", "legion_hexapod", "starfish"
]

max_parallel = 4
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
      "20", "--generations", "20"
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

import os
import subprocess
import time

species_list = [
    "quadruped", "goliath_crawler", "legion_hexapod", "aegis_turtle",
    "ein_corgi", "khepri_beetle", "giraffe_default", "arachne_spider",
    "centipede", "scorpion", "gorilla", "starfish", "snake",
    "kangaroo", "crab", "megapede", "stilts_biped"
]

max_parallel = 5
active_processes = []
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
  p = subprocess.Popen([".venv/bin/python3", "auto_evolve.py", "--species", species])
  active_processes.append(p)
  
# Wait for all to finish
while active_processes:
  for p in active_processes:
    if p.poll() is not None:
      active_processes.remove(p)
  time.sleep(1)
  
print("All species evolved!")

import math
import os
import random
import time

import mujoco
import mujoco.viewer
import yaml

from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator


def get_top_candidates(num_candidates=5):
  """Fetches top candidates from LEADERBOARD.md."""
  candidates = []
  if os.path.exists("LEADERBOARD.md"):
    with open("LEADERBOARD.md", "r") as f:
      for line in f:
        if line.startswith("|") and not line.startswith(
            "| Rank") and not line.startswith("|---"):
          parts = line.split("|")
          if len(parts) > 2:
            name = parts[2].strip()
            if name:
              if not name.endswith(".yaml"):
                name += ".yaml"
              candidates.append(name)
              if len(candidates) >= num_candidates:
                break
  return candidates


def run_demo():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-viewer",
                      action="store_true",
                      help="Run without GUI viewer")
  parser.add_argument("--record",
                      action="store_true",
                      help="Record frames to results")
  parser.add_argument("--top-candidates",
                      type=int,
                      default=5,
                      help="Number of top candidates from leaderboard to load")
  args = parser.parse_args()

  env = Environment()
  env.rough_terrain = False  # Disable rough terrain for stability.
  env.wind = [0.0, 0.0, 0.0]  # Disable wind for demo stability.
  env.floor_size = [15.0, 15.0,
                    0.05]  # Floor is 30x30m, fitting the 24m ring with edges.

  # Add 10 static blocks
  for i in range(10):
    env.obstacles.append({
        "type": "box",
        "size": [0.3, 0.3, 0.3],
        "pos": [random.uniform(-8.0, 8.0),
                random.uniform(-8.0, 8.0), 0.3],
        "color": [0.5, 0.5, 0.5, 1.0]
    })

  # Add 2 rolling spheres for minimal chaos.
  for i in range(2):
    env.obstacles.append({
        "type": "sphere",
        "size": [0.5, 0.5, 0.5],
        "pos": [random.uniform(-3.0, 3.0),
                random.uniform(-3.0, 3.0), 3.0],
        "mass": 20.0,
        "color": [0.3, 0.3, 0.3, 1.0]
    })

  # Giant boulder removed to prevent physics freeze.

  # Load some best agents.
  agents = []
  agents_dir = "templates/agents"
  if os.path.exists(agents_dir):
    # Get all evolved agents (excluding legacy quadruped biped files.)
    files = [f for f in os.listdir(agents_dir) if f.endswith(".yaml")]
    random.shuffle(files)

    # Fetch top candidates dynamically from leaderboard.
    top_candidates = get_top_candidates(args.top_candidates)
    selected_files = []
    for f in top_candidates:
      if os.path.exists(os.path.join(agents_dir, f)):
        selected_files.append(f)

    for f in files:
      if f not in selected_files:
        selected_files.append(f)
        if len(selected_files) >= 15:
          break

    # Load up to 15 agents for maximum diversity.
    for i, f in enumerate(selected_files):
      with open(os.path.join(agents_dir, f), "r") as file:
        cfg = yaml.safe_load(file)["agents"][0]

        agent_type = cfg.get("type", "default")
        template_path = f"templates/agents/{agent_type}.yaml"

        # Add index to name to avoid collisions.
        agent_name = f"{cfg['name']}_{i}"

        if os.path.exists(template_path):
          with open(template_path, "r") as tf:
            template_cfg = yaml.safe_load(tf)
            merged_cfg = {**template_cfg["agents"][0], **cfg}
            agent = ConfigurableAgent(name=agent_name, config=merged_cfg)
        else:
          agent = ConfigurableAgent(name=agent_name, config=cfg)

        # Heavily increase amplitude to force aggressive movement.
        agent.amplitude = max(2.0, agent.amplitude * 2.0)
        # Speed up gait frequency to make them move faster.
        agent.frequency *= 1.5
        # Force longer legs to prevent stumpy non-moving agents.
        agent.leg_length_scale = max(1.5, agent.leg_length_scale)

        agents.append(agent)

  if not agents:
    print(
        "No evolved agents found in templates/agents/. Please run auto_evolve.py first."
    )
    return

  # Randomly scatter agents on the floor with collision avoidance.
  min_dist = 2.0
  for agent in agents:
    placed = False
    for _ in range(100):
      pos = [random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), 0.5]
      too_close = False
      for other in agents:
        if other != agent and hasattr(other, "pos"):
          d = ((pos[0] - other.pos[0])**2 + (pos[1] - other.pos[1])**2)**0.5
          if d < min_dist:
            too_close = True
            break
      if not too_close:
        agent.pos = pos
        placed = True
        break
    if not placed:
      print(
          f"Warning: Could not find non-overlapping position for {agent.name}.")
      agent.pos = [
          random.uniform(-10.0, 10.0),
          random.uniform(-10.0, 10.0), 0.5
      ]

  orch = Orchestrator(env, agents, death_threshold=float('inf'))
  orch.initialize()

  # Add 20 more food items to fill the arena (50 total).
  for _ in range(20):
    orch.food_positions.append(
        [random.uniform(-15.0, 15.0),
         random.uniform(-15.0, 15.0), 0.1])

  record_dir = None
  renderer = None
  if args.record:
    timestamp = int(time.time())
    # Use double underscore to match evolution GIF grouper
    record_dir = f"results/demo__{timestamp}"
    os.makedirs(record_dir, exist_ok=True)
    print(f"Recording frames to {record_dir}...")
    renderer = mujoco.Renderer(orch.model, 1024, 1024)

  # Spatially distribute agents and apply random yaw rotations.
  for i, agent in enumerate(agents):
    try:
      jnt_id = mujoco.mj_name2id(orch.model, mujoco.mjtObj.mjOBJ_JOINT,
                                 agent.name)
      if jnt_id >= 0:
        qpos_addr = orch.model.jnt_qposadr[jnt_id]
        # Apply position from agent.pos (set earlier in circle)
        orch.data.qpos[qpos_addr:qpos_addr + 3] = agent.pos

        # Apply random yaw rotation (quaternion [cos(t/2), 0, 0, sin(t/2)])
        theta = random.uniform(0, 2 * math.pi)
        orch.data.qpos[qpos_addr + 3:qpos_addr + 7] = [
            math.cos(theta / 2), 0.0, 0.0,
            math.sin(theta / 2)
        ]
    except Exception as e:
      print(f"Failed to set initial pose for {agent.name}: {e}")

  # Enable synthesis in GUI mode as requested! (May cause brief freezes on reload).
  orch.enable_synthesis = True
  orch.enable_food = True if args.no_viewer else False
  orch.enable_flip_death = True  # Enable flip death (health drain)!
  orch.enable_respawn = False  # Disable respawn for demo stability!
  orch.enable_event_logging = True  # Enable event logging for demo!.

  paused = False

  # Run 10 physics steps per render frame to speed up simulation without starving UI thread.
  steps_per_frame = 10

  def key_callback(keycode):
    nonlocal paused, steps_per_frame
    try:
      key = chr(keycode)
      if key == ' ':
        paused = not paused
        print(f"Simulation {'paused' if paused else 'resumed'}")
      elif key in ('g', 'G'):
        curr_grav = orch.env.gravity
        new_grav = [curr_grav[0], curr_grav[1], -curr_grav[2]]
        orch.env.set_gravity(new_grav)
        orch.update_physics()
      elif key in ('w', 'W'):
        curr_wind = orch.env.wind
        new_wind = [curr_wind[0] + 1.0, curr_wind[1], curr_wind[2]]
        orch.env.wind = new_wind
        orch.update_physics()
        print(f"Increased wind to: {new_wind}")
      elif key in ('s', 'S'):
        curr_wind = orch.env.wind
        new_wind = [curr_wind[0] - 1.0, curr_wind[1], curr_wind[2]]
        orch.env.wind = new_wind
        orch.update_physics()
        print(f"Decreased wind to: {new_wind}")
      elif key in ('r', 'R'):
        for agent in orch.agents:
          orch._respawn_agent(agent)
        print("Respawned all agents!")
      elif key in ('=', '+'):
        steps_per_frame += 1
        print(f"Increased simulation speed to {steps_per_frame}x")
      elif key in ('-', '_'):
        steps_per_frame = max(1, steps_per_frame - 1)
        print(f"Decreased simulation speed to {steps_per_frame}x")
    except ValueError:
      pass

  print("Starting demo window... It will auto-close in 2 minutes.")
  print(
      "Controls: Space to pause, G to invert gravity, W/S to adjust wind, R to respawn all."
  )

  if args.no_viewer:
    print("Running without viewer...")
    start_time = time.time()
    while time.time() - start_time < 10:
      orch.step()
      time.sleep(orch.model.opt.timestep)
    print("Demo finished.")
    return

  with mujoco.viewer.launch_passive(orch.model,
                                    orch.data,
                                    key_callback=key_callback) as viewer:
    # Set camera to free but positioned at our good birds-eye view
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = [0.0, 0.0, 0.0]
    viewer.cam.distance = env.camera_distance
    viewer.cam.elevation = env.camera_elevation
    viewer.cam.azimuth = env.camera_azimuth

    start_time = time.time()
    frame_count = 0
    try:
      while viewer.is_running() and time.time() - start_time < 120:
        if not paused:
          step_start = time.time()

          for _ in range(steps_per_frame):
            orch.step()
          viewer.sync()

          # Record every 5th frame (approx 20 fps) from user's POV to prevent CPU hogging.
          if record_dir and renderer and frame_count % 5 == 0:
            renderer.update_scene(orch.data, camera=viewer.cam)
            pixels = renderer.render()
            frame_idx = int(orch.data.time / orch.model.opt.timestep)
            frame_path = os.path.join(record_dir, f"frame_{frame_idx:05d}.ppm")
            with open(frame_path, "wb") as f:
              f.write(f"P6\n1024 1024\n255\n".encode())
              f.write(pixels.tobytes())

          frame_count += 1

          # Sleep for 10ms to prevent CPU hogging and UI thread starvation.
          time.sleep(0.01)
        else:
          time.sleep(0.1)  # Sleep longer if paused.
    except KeyboardInterrupt:
      print("\nInterrupted by user.")
      if record_dir:
        print("Generating GIF from recorded frames...")
        from create_evolution_gif import create_gif
        create_gif(species_filter="demo")

  print(f"Demo finished in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
  run_demo()

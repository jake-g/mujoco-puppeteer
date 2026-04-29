"""CLI for selecting and launching MuJoCo simulation templates."""

import argparse
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET

import mujoco.viewer
import yaml

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def list_templates(templates_dir: str):
  """Lists available templates in the directory."""
  print("Available Templates:")
  for root, _, files in os.walk(templates_dir):
    for filename in files:
      if filename.endswith(".yaml"):
        rel_path = os.path.relpath(os.path.join(root, filename), templates_dir)
        print(f"  - {rel_path[:-5]}")


def load_config(path: str) -> dict:
  """Loads a YAML config file."""
  with open(path, "r") as f:
    return yaml.safe_load(f)


def launch_simulation(config: dict):
  """Launches the simulation based on the config."""
  env = Environment()

  # Apply environment config
  if "environment" in config:
    env_cfg = config["environment"]
    if "gravity" in env_cfg:
      env.set_gravity(env_cfg["gravity"])
    if "floor_size" in env_cfg:
      env.floor_size = env_cfg["floor_size"]
    if "floor_rgb1" in env_cfg:
      env.floor_rgb1 = env_cfg["floor_rgb1"]
    if "floor_rgb2" in env_cfg:
      env.floor_rgb2 = env_cfg["floor_rgb2"]
    if "sky_rgb1" in env_cfg:
      env.sky_rgb1 = env_cfg["sky_rgb1"]
    if "sky_rgb2" in env_cfg:
      env.sky_rgb2 = env_cfg["sky_rgb2"]
    if "obstacles" in env_cfg:
      env.obstacles = env_cfg["obstacles"]
    if "camera" in env_cfg:
      env.camera_pos = env_cfg["camera"].get("pos", env.camera_pos)
      env.camera_xyaxes = env_cfg["camera"].get("xyaxes", env.camera_xyaxes)
      env.camera_distance = env_cfg["camera"].get("distance",
                                                  env.camera_distance)
      env.camera_elevation = env_cfg["camera"].get("elevation",
                                                   env.camera_elevation)
      env.camera_azimuth = env_cfg["camera"].get("azimuth", env.camera_azimuth)
    if "rough_terrain" in env_cfg:
      env.rough_terrain = env_cfg["rough_terrain"]

  agents = []
  if "agents" in config:
    for agent_cfg in config["agents"]:
      size_scale = agent_cfg.get("size_scale", 1.0)
      agent_type = agent_cfg.get("type", "default")

      agent: Agent
      template_path = f"templates/agents/{agent_type}_default.yaml"
      if not os.path.exists(template_path):
        template_path = f"templates/agents/{agent_type}.yaml"

      if os.path.exists(template_path):
        with open(template_path, "r") as f:
          template_cfg = yaml.safe_load(f)
          # Merge template config with instance config
          merged_cfg = {**template_cfg["agents"][0], **agent_cfg}
          agent = ConfigurableAgent(name=agent_cfg["name"],
                                    size_scale=size_scale,
                                    config=merged_cfg)
      elif "limbs" in agent_cfg:
        agent = ConfigurableAgent(name=agent_cfg["name"],
                                  size_scale=size_scale,
                                  config=agent_cfg)
      else:
        agent = Agent(name=agent_cfg["name"], size_scale=size_scale)
      if "pos" in agent_cfg:
        agent.pos = agent_cfg["pos"]
      if "color" in agent_cfg:
        agent.color = agent_cfg["color"]
      if "frequency" in agent_cfg:
        agent.frequency = agent_cfg["frequency"]
      if "phase" in agent_cfg:
        agent.phase = agent_cfg["phase"]
      agents.append(agent)

  death_threshold = config.get("death_threshold", 3.0)
  orchestrator = Orchestrator(env, agents, death_threshold=death_threshold)
  orchestrator.initialize()

  logger.info("Starting visual simulation from CLI...")
  logger.info("Use the UI panels to inspect and edit parameters.")

  paused = False

  def key_callback(keycode):
    nonlocal paused
    try:
      key = chr(keycode)
      if key == ' ':
        paused = not paused
        logger.info("Simulation %s", "paused" if paused else "resumed")
      elif key in ('g', 'G'):
        # Invert gravity
        curr_grav = orchestrator.env.gravity
        new_grav = [curr_grav[0], curr_grav[1], -curr_grav[2]]
        orchestrator.env.set_gravity(new_grav)
        orchestrator.update_physics()
      elif key in ('w', 'W'):
        # Increase wind.
        curr_wind = orchestrator.env.wind
        new_wind = [curr_wind[0] + 1.0, curr_wind[1], curr_wind[2]]
        orchestrator.env.wind = new_wind
        orchestrator.update_physics()
        logger.info("Increased wind to: %s", new_wind)
      elif key in ('s', 'S'):
        # Decrease wind.
        curr_wind = orchestrator.env.wind
        new_wind = [curr_wind[0] - 1.0, curr_wind[1], curr_wind[2]]
        orchestrator.env.wind = new_wind
        orchestrator.update_physics()
        logger.info("Decreased wind to: %s", new_wind)
      elif key in ('r', 'R'):
        # Respawn all agents.
        for agent in orchestrator.agents:
          orchestrator._respawn_agent(agent)
        logger.info("Respawned all agents!")
    except ValueError:
      # Handle special keys or non-printable characters
      pass

  # On Mac, launch_passive is often more robust when run via mjpython
  with mujoco.viewer.launch_passive(orchestrator.model,
                                    orchestrator.data,
                                    key_callback=key_callback) as viewer:
    # Set camera to free but positioned at our good birds-eye view
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = [0.0, 0.0, 0.0]
    viewer.cam.distance = orchestrator.env.camera_distance
    viewer.cam.elevation = orchestrator.env.camera_elevation
    viewer.cam.azimuth = orchestrator.env.camera_azimuth

    while viewer.is_running():
      if not paused:
        step_start = time.time()
        orchestrator.step()
        viewer.sync()

        assert orchestrator.model is not None
        time_until_next_step = orchestrator.model.opt.timestep - (time.time() -
                                                                  step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
      else:
        time.sleep(0.1)


def main():
  parser = argparse.ArgumentParser(description="MuJoCo Puppeteer CLI")
  parser.add_argument("--list",
                      action="store_true",
                      help="List available templates")
  parser.add_argument("--run", type=str, help="Run a specific template by name")

  args = parser.parse_args()

  current_dir = os.path.dirname(os.path.abspath(__file__))
  templates_dir = os.path.join(current_dir, "templates")

  if args.list:
    list_templates(templates_dir)
    return

  if args.run:
    template_path = None
    for root, _, files in os.walk(templates_dir):
      if f"{args.run}.yaml" in files:
        template_path = os.path.join(root, f"{args.run}.yaml")
        break

    if not template_path:
      print(f"Error: Template '{args.run}' not found.")
      list_templates(templates_dir)
      return

    config = load_config(template_path)
    launch_simulation(config)
    return

  parser.print_help()


if __name__ == "__main__":
  main()

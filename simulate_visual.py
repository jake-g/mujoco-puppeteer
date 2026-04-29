"""Visualization script for MuJoCo simulation."""

import logging
import time
import mujoco
import mujoco.viewer
from environment import Environment
from agent import Agent
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
  # Initialize environment and agents
  env = Environment()
  agent1 = Agent(name="agent_1")
  agent1.pos = [0.0, 0.0, 1.0]

  agent2 = Agent(name="agent_2")
  agent2.pos = [1.0, 1.0, 1.0]

  orchestrator = Orchestrator(env, [agent1, agent2])
  orchestrator.initialize()

  logger.info("Starting visual simulation...")

  # On Mac, this script MUST be run with 'mjpython'
  # launch_passive requires the main thread to handle rendering
  with mujoco.viewer.launch_passive(
      orchestrator.model, orchestrator.data
  ) as viewer:
    # Close the viewer automatically after 30 wall-seconds for this demo
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
      step_start = time.time()

      # Step simulation
      orchestrator.step()

      # Example interaction: invert gravity every 5 seconds
      if (
          int(orchestrator.data.time) % 10 >= 5
          and int(orchestrator.data.time) > 0
      ):
        env.set_gravity([0.0, 0.0, 5.0])  # Invert gravity (push up)
      else:
        env.set_gravity([0.0, 0.0, -9.81])

      orchestrator.update_physics()

      # Sync viewer
      viewer.sync()

      # Time keeping (crude)
      time_until_next_step = orchestrator.model.opt.timestep - (
          time.time() - step_start
      )
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)


if __name__ == "__main__":
  main()

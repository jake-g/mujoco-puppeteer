"""Genetic Algorithm for evolving agent walking policies."""

import logging
import os
import random
import time
import yaml
from environment import Environment
from agent import QuadrupedAgent
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_population(agents: list[QuadrupedAgent],
                        duration: float = 5.0) -> list[tuple[QuadrupedAgent, float]]:
  """Evaluates a population of agents in the simulation.

  Args:
      agents: List of agents to evaluate.
      duration: Simulation duration in seconds.

  Returns:
      list: Sorted list of (agent, reward) tuples.
  """
  env = Environment()
  env.floor_size = [50.0, 50.0, 0.05]

  # Space them out to avoid collisions during training
  for i, agent in enumerate(agents):
    agent.pos = [0.0, i * 2.0, 1.0]

  orchestrator = Orchestrator(env, agents, death_threshold=float('inf'))
  orchestrator.initialize()

  # Run simulation headless
  steps = int(duration / orchestrator.model.opt.timestep)
  logger.info("Running simulation for %d steps...", steps)

  for _ in range(steps):
    orchestrator.step()

  # Calculate final rewards (distance in x)
  results = []
  for agent in agents:
    try:
      body = orchestrator.data.body(agent.name)
      # Reward is x position (distance traveled forward)
      reward = body.xpos[0]
      results.append((agent, reward))
    except Exception as e:
      logger.error("Failed to get result for %s: %s", agent.name, e)
      results.append((agent, -100.0))

  # Sort by reward descending
  results.sort(key=lambda x: x[1], reverse=True)
  return results


def main():
  pop_size = 10
  generations = 20
  eval_duration = 5.0

  # Initialize population
  population = [
      QuadrupedAgent(name=f"agent_{i}") for i in range(pop_size)
  ]

  for gen in range(generations):
    logger.info("=== Generation %d ===", gen)

    # Evaluate
    results = evaluate_population(population, duration=eval_duration)

    logger.info("Top rewards:")
    for i in range(min(3, len(results))):
      logger.info("  %s: %.2f", results[i][0].name, results[i][1])

    best_agent = results[0][0]
    logger.info("Best agent frequency: %.2f, phase: %.2f",
                best_agent.frequency, best_agent.phase)

    # Selection & Breeding
    top_performers = [results[i][0] for i in range(2)]  # Top 2

    new_population = []
    # Keep the best one (elitism)
    best_agent.name = "agent_0"
    new_population.append(best_agent)

    # Breed the rest
    for i in range(1, pop_size):
      parent1 = random.choice(top_performers)
      parent2 = random.choice(top_performers)

      # Create new agent
      child = QuadrupedAgent(name=f"agent_{i}")

      # Mix parameters
      child.frequency = (parent1.frequency + parent2.frequency) / 2
      child.phase = (parent1.phase + parent2.phase) / 2
      child.phase_offsets = [(p1 + p2) / 2 for p1, p2 in zip(parent1.phase_offsets, parent2.phase_offsets)]

      # Mutate (Multiplicative as seen in evolution-sim)
      child.frequency = child.frequency + child.frequency * random.uniform(-0.1, 0.1)
      child.phase = child.phase + child.phase * random.uniform(-0.1, 0.1)
      child.phase_offsets = [p + p * random.uniform(-0.1, 0.1) for p in child.phase_offsets]

      new_population.append(child)

    population = new_population

  # Save best result to a template
  best_agent = results[0][0]
  best_config = {
      "environment": {
          "floor_size": [20.0, 20.0, 0.05],
          "floor_rgb1": [0.0, 0.0, 0.0],
          "floor_rgb2": [1.0, 1.0, 1.0],
          "sky_rgb1": [0.6, 0.8, 1.0],
          "sky_rgb2": [1.0, 1.0, 1.0]
      },
      "agents": [{
          "name": "evolved_quad",
          "type": "quadruped",
          "pos": [0.0, 0.0, 1.0],
          "color": [0.0, 1.0, 0.0, 1.0],
          "size_scale": 1.0,
          "frequency": float(best_agent.frequency),
          "phase": float(best_agent.phase)
      }]
  }

  os.makedirs("templates", exist_ok=True)
  with open("templates/evolved_quadruped.yaml", "w") as f:
    yaml.dump(best_config, f)

  logger.info("Saved best template to templates/evolved_quadruped.yaml")


if __name__ == "__main__":
  main()

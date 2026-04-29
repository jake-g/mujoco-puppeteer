"""Automated evolution system for multiple agent types."""

import logging
import os
import random
from typing import Sequence

import yaml

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_population(agents: Sequence[Agent],
                        duration: float = 5.0) -> list[tuple[Agent, float]]:
  """Evaluates a population of agents in the simulation."""
  env = Environment()
  env.floor_size = [50.0, 50.0, 0.05]

  # Space them out
  for i, agent in enumerate(agents):
    agent.pos = [0.0, i * 2.0, 1.0]

  # Disable death on fall during training
  orchestrator = Orchestrator(env, agents, death_threshold=float('inf'))
  orchestrator.initialize()

  assert orchestrator.model is not None
  steps = int(duration / orchestrator.model.opt.timestep)

  for _ in range(steps):
    orchestrator.step()

  results = []
  for agent in agents:
    try:
      reward = agent.reward
      results.append((agent, reward))
    except Exception as e:
      logger.error("Failed to get result for %s: %s", agent.name, e)
      results.append((agent, -100.0))

  results.sort(key=lambda x: x[1], reverse=True)
  return results


def tournament_selection(results, k=3):
  """Picks the best of k random individuals."""
  indices = random.sample(range(len(results)), k)
  best_idx = min(indices)
  return results[best_idx][0]


def evolve_species(agent_class,
                   species_name,
                   config,
                   pop_size=10,
                   generations=5):
  """Runs GA for a specific agent class."""
  logger.info("=== Evolving Species: %s ===", species_name)

  population = [
      agent_class(name=f"{species_name}_{i:04x}", config=config)
      for i in range(pop_size)
  ]

  for gen in range(generations):
    results = evaluate_population(population)
    logger.info("Gen %d Best Reward: %.2f", gen, results[0][1])

    best_agent = results[0][0]
    top_performers = [results[i][0] for i in range(2)]

    new_population = []
    best_agent.name = f"{species_name}_0000"
    new_population.append(best_agent)

    for i in range(1, pop_size):
      parent1 = tournament_selection(results, k=3)
      parent2 = tournament_selection(results, k=3)

      child = agent_class(name=f"{species_name}_{i:04x}", config=config)

      # Crossover
      genes1 = [
          parent1.frequency, parent1.phase, parent1.amplitude,
          parent1.leg_length_scale
      ] + parent1.phase_offsets
      genes2 = [
          parent2.frequency, parent2.phase, parent2.amplitude,
          parent2.leg_length_scale
      ] + parent2.phase_offsets

      crossover_point = random.randint(1, 7)
      child_genes = genes1[:crossover_point] + genes2[crossover_point:]

      child.frequency = child_genes[0]
      child.phase = child_genes[1]
      child.amplitude = child_genes[2]
      child.leg_length_scale = child_genes[3]
      child.phase_offsets = child_genes[4:]

      # Mutate
      child.frequency = child.frequency + child.frequency * random.uniform(
          -0.1, 0.1)
      child.phase = child.phase + child.phase * random.uniform(-0.1, 0.1)
      child.amplitude = child.amplitude + child.amplitude * random.uniform(
          -0.1, 0.1)
      child.leg_length_scale = child.leg_length_scale + child.leg_length_scale * random.uniform(
          -0.1, 0.1)
      child.phase_offsets = [
          p + p * random.uniform(-0.1, 0.1) for p in child.phase_offsets
      ]
      child.update_id()

      new_population.append(child)

    population = new_population

  # Save best to template
  best_agent = results[0][0]
  current_gen = config.get("generation", 0)
  total_gen = current_gen + generations

  # Rename with structured format!
  best_agent.name = f"{species_name}__{best_agent.id}__gen{total_gen}"

  save_config = {
      "environment": {
          "floor_size": [20.0, 20.0, 0.05],
          "floor_rgb1": [0.0, 0.0, 0.0],
          "floor_rgb2": [1.0, 1.0, 1.0],
      },
      "agents": [{
          "id": best_agent.id,
          "name": best_agent.name,
          "type": species_name,
          "generation": total_gen,
          "pos": [0.0, 0.0, 1.0],
          "color": [0.0, 1.0, 0.0, 1.0],
          "size_scale": float(best_agent.size_scale),
          "frequency": float(best_agent.frequency),
          "phase": float(best_agent.phase),
          "amplitude": float(best_agent.amplitude),
          "leg_length_scale": float(best_agent.leg_length_scale),
          "phase_offsets": [float(p) for p in best_agent.phase_offsets],
          "parent_ids": getattr(best_agent, "parent_ids", []),
      }],
  }

  os.makedirs("templates/agents", exist_ok=True)
  filename = f"templates/agents/{best_agent.name}.yaml"
  with open(filename, "w") as f:
    yaml.dump(save_config, f)

  logger.info("Saved best template to %s", filename)


def evaluate_template(template_path: str) -> tuple[float, int]:
  """Evaluates a template and returns (score, steps)."""
  with open(template_path, "r") as f:
    config = yaml.safe_load(f)

  env = Environment()
  if "environment" in config:
    env_cfg = config["environment"]
    if "floor_size" in env_cfg:
      env.floor_size = env_cfg["floor_size"]
    if "obstacles" in env_cfg:
      env.obstacles = env_cfg["obstacles"]

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
          # Merge template config with instance config!
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

      if "frequency" in agent_cfg:
        agent.frequency = agent_cfg["frequency"]
      if "phase" in agent_cfg:
        agent.phase = agent_cfg["phase"]
      if "amplitude" in agent_cfg:
        agent.amplitude = agent_cfg["amplitude"]
      if "leg_length_scale" in agent_cfg:
        agent.leg_length_scale = agent_cfg["leg_length_scale"]
      if "phase_offsets" in agent_cfg:
        agent.phase_offsets = agent_cfg["phase_offsets"]

      agents.append(agent)

  death_threshold = config.get("death_threshold", 3.0)
  orchestrator = Orchestrator(env, agents, death_threshold=death_threshold)
  orchestrator.initialize()

  assert orchestrator.model is not None
  steps = int(10.0 / orchestrator.model.opt.timestep)
  for _ in range(steps):
    orchestrator.step()

  total_reward = 0.0
  total_steps = 0
  for agent in agents:
    total_reward += agent.reward
    total_steps += agent.steps

  avg_reward = total_reward / len(agents) if agents else 0.0
  return avg_reward, total_steps


def update_leaderboard():
  """Evaluates all templates and updates the leaderboard."""
  results = []
  folders = ["templates/scenes", "templates/agents"]

  for folder in folders:
    if not os.path.exists(folder):
      continue
    for filename in os.listdir(folder):
      if filename.endswith(".yaml"):
        path = os.path.join(folder, filename)
        logger.info("Evaluating %s...", filename)
      try:
        score, steps = evaluate_template(path)
        results.append({"name": filename[:-5], "score": score, "steps": steps})
      except Exception as e:
        logger.error("Failed to evaluate %s: %s", filename, e)

  # Sort by score descending
  results.sort(key=lambda x: x["score"], reverse=True)

  # Generate Markdown
  markdown = "# Simulation Leaderboard\n\n"
  markdown += "| Rank | Config | Score (Reward) | Total Steps |\n"
  markdown += "|------|--------|----------------|-------------|\n"

  for i, res in enumerate(results):
    markdown += f"| {i+1} | {res['name']} | {res['score']:.2f} | {res['steps']} |\n"

  with open("LEADERBOARD.md", "w") as f:
    f.write(markdown)

  logger.info("Leaderboard updated in LEADERBOARD.md")


def main():
  # Evolve Quadruped (Using base Agent as default)
  evolve_species(Agent, "quadruped", {}, pop_size=20, generations=20)

  # Evolve Goliath (Crawler)
  with open("templates/agents/goliath_crawler.yaml", "r") as f:
    crawler_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "goliath_crawler",
                 crawler_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Legion (Hexapod)
  with open("templates/agents/legion_hexapod.yaml", "r") as f:
    hexapod_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "legion_hexapod",
                 hexapod_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Aegis (Turtle)
  with open("templates/agents/aegis_turtle.yaml", "r") as f:
    turtle_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "aegis_turtle",
                 turtle_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Ein (Corgi)
  with open("templates/agents/ein_corgi.yaml", "r") as f:
    corgi_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "ein_corgi",
                 corgi_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Khepri (Beetle)
  with open("templates/agents/khepri_beetle.yaml", "r") as f:
    khepri_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "khepri_beetle",
                 khepri_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Giraffe
  with open("templates/agents/giraffe_default.yaml", "r") as f:
    giraffe_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "giraffe_default",
                 giraffe_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Arachne (Spider)
  with open("templates/agents/arachne_spider.yaml", "r") as f:
    arachne_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "arachne_spider",
                 arachne_cfg,
                 pop_size=20,
                 generations=20)

  # Update Leaderboard!
  update_leaderboard()


if __name__ == "__main__":
  main()

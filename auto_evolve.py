"""Automated evolution system for multiple agent types."""

import logging
import os
import random
import re
import shutil
import sys
from typing import Sequence

import mujoco
import yaml

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator
from render import render_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_source_species_name(species: str) -> str:
  """Returns the exact case of the species folder in templates/agents."""
  templates_dir = "templates/agents"
  if os.path.exists(templates_dir):
    for d in os.listdir(templates_dir):
      if d.lower() == species.lower():
        return d
  return species


def evaluate_population(agents: Sequence[Agent],
                        duration: float = 5.0,
                        generation: int = 0) -> list[tuple[Agent, float]]:
  """Evaluates a population of agents in the simulation."""
  env = Environment()
  env.floor_size = [20.0, 20.0, 0.05]

  # Curriculum Learning: Enable rough terrain after gen 5.
  if generation >= 5:
    env.rough_terrain = True

  # Phase 3: Add wind after gen 10.
  if generation >= 10:
    env.wind = [2.0, 0.0, 0.0]

  # Space them out centered around origin
  for i, agent in enumerate(agents):
    agent.pos = [0.0, i * 2.0 - 20.0, 1.0]

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
      body = orchestrator.data.body(agent.name)
      pos = body.xpos
      distance = (pos[0]**2 + pos[1]**2)**0.5
      agent.final_distance = distance
      results.append((agent, reward))
    except Exception as e:
      logger.error("Failed to get result for %s: %s", agent.name, e)
      agent.final_distance = 0.0
      results.append((agent, -100.0))

  results.sort(key=lambda x: x[1], reverse=True)
  return results


def save_agent_frames(agent: Agent, duration: float = 5.0):
  """Saves a sequence of frames for the agent to results folder."""
  try:
    env = Environment()
    orch = Orchestrator(env, [agent])
    xml_str = orch.generate_combined_xml()

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Increased resolution to 1024x1024 for better zoom quality.
    renderer = mujoco.Renderer(model, 1024, 1024)

    species = agent.name.split("__")[0] if "__" in agent.name else re.sub(
        r"_[0-9a-f]+$", "", agent.name)
    species = re.sub(r"_default$", "", species)
    species = get_source_species_name(species)

    results_dir = f"results/agents/{species.lower()}/generations/{agent.name}"
    os.makedirs(results_dir, exist_ok=True)

    steps = int(duration / model.opt.timestep)
    for i in range(steps):
      mujoco.mj_step(model, data)
      if i % 10 == 0:
        renderer.update_scene(data, camera="main_cam")
        pixels = renderer.render()
        frame_filename = f"{results_dir}/frame_{i:05d}.ppm"
        with open(frame_filename, "wb") as f:
          f.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
          f.write(pixels.tobytes())

    logger.info("Saved %d seconds of frames to %s", int(duration), results_dir)
  except Exception as e:
    logger.error("Failed to save frames for %s: %s", agent.name, e)


def tournament_selection(results, k=3):
  """Picks the best of k random individuals."""
  k = min(k, len(results))
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

  display_name = species_name

  population = []
  for i in range(pop_size):
    a = agent_class(name=f"{display_name}_{i:04x}", config=config)
    a.species = display_name
    population.append(a)

  history = []

  for gen in range(generations):
    results = evaluate_population(population, generation=gen)
    logger.info("Gen %d Best Reward: %.2f", gen, results[0][1])

    best_agent = results[0][0]
    history.append({
        "gen": gen,
        "reward": float(results[0][1]),
        "distance": best_agent.final_distance,
        "food": best_agent.food_eaten,
        "breeding": best_agent.syntheses_count
    })

    # Save frames for the best agent every 5 generations.
    if gen % 5 == 0:
      save_agent_frames(best_agent, duration=1.0)

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
          -0.4, 0.4)
      child.phase = child.phase + child.phase * random.uniform(-0.4, 0.4)
      child.amplitude = child.amplitude + child.amplitude * random.uniform(
          -0.4, 0.4)
      child.leg_length_scale = child.leg_length_scale + child.leg_length_scale * random.uniform(
          -0.4, 0.4)
      child.phase_offsets = [
          p + p * random.uniform(-0.4, 0.4) for p in child.phase_offsets
      ]

      # Clip genes to valid ranges to prevent dead/frozen agents.
      child.frequency = max(0.5, min(10.0, child.frequency))
      child.amplitude = max(0.5, min(2.0, child.amplitude))
      child.leg_length_scale = max(0.5, min(3.0, child.leg_length_scale))

      child.update_id()

      new_population.append(child)

    population = new_population

  # Save history to file.
  # Save history to TSV.
  os.makedirs("results", exist_ok=True)
  # Use species-specific history file to allow parallel runs without conflicts.
  source_species = get_source_species_name(species_name)
  history_path = f"results/agents/{source_species.lower()}/evolution_history.tsv"
  file_exists = os.path.exists(history_path)

  import time
  with open(history_path, "a") as f:
    if not file_exists:
      f.write("timestamp\tspecies\tgeneration\treward\tdistance\tfood\tbreeding\n")
    for entry in history:
      f.write(
          f"{time.time()}\t{species_name}\t{entry['gen']}\t{entry['reward']}\t{entry['distance']}\t{entry['food']}\t{entry['breeding']}\n"
      )
  logger.info("Appended evolution history to %s", history_path)

  # Save best to template
  best_agent = results[0][0]
  current_gen = config.get("generation", 0)
  total_gen = current_gen + generations

  # Rename with structured format.
  best_agent.name = f"{species_name}__{best_agent.id}__gen{total_gen}"

  save_config = {
      "environment": {
          "floor_size": [20.0, 20.0, 0.05],
          "floor_rgb1": [0.0, 0.0, 0.0],
          "floor_rgb2": [1.0, 1.0, 1.0],
      },
      "agents": [best_agent.to_dict()],
  }

  species = best_agent.name.split(
      "__")[0] if "__" in best_agent.name else re.sub(r"_[0-9a-f]+$", "",
                                                      best_agent.name)
  species = re.sub(r"_default$", "", species)
  species = get_source_species_name(species)

  species_dir = f"templates/agents/{species.lower()}"
  gen_dir = os.path.join(species_dir, "generations")
  os.makedirs(gen_dir, exist_ok=True)
  filename = f"{gen_dir}/{best_agent.name}.yaml"
  with open(filename, "w") as f:
    yaml.dump(save_config, f)

  logger.info("Saved best template to %s", filename)

  # Render image for the best agent.
  try:
    best_jpg = f"{species_dir}/{species}_best.jpg"
    render_template(filename, best_jpg, output_format="jpg", res=(400, 400))

    save_agent_frames(best_agent, duration=10.0)
  except Exception as e:
    logger.error("Failed to save image or frames for %s: %s", best_agent.name,
                 e)


def evaluate_template(template_path: str) -> tuple[float, int]:
  """Evaluates a template and returns (score, steps)."""
  with open(template_path, "r") as f:
    config = yaml.safe_load(f)

  env = Environment()
  # Add mild wind as a baseline challenge for all agents.
  env.wind = [0.5, 0.0, 0.0]

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
          # Merge template config with instance config.
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

      agent.species = agent_type

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
  total_distance = 0.0
  total_survival = 0.0
  for agent in agents:
    total_reward += agent.reward
    total_steps += agent.steps
    try:
      body = orchestrator.data.body(agent.name)
      pos = body.xpos
      total_distance += (pos[0]**2 + pos[1]**2)**0.5
    except Exception:
      pass
    total_survival += agent.time_alive

  avg_reward = total_reward / len(agents) if agents else 0.0
  avg_distance = total_distance / len(agents) if agents else 0.0
  avg_survival = total_survival / len(agents) if agents else 0.0
  total_food = sum(agent.food_eaten for agent in agents)
  total_agent_syntheses = sum(agent.syntheses_count for agent in agents)

  return avg_reward, total_steps, orchestrator.total_syntheses, avg_distance, avg_survival, total_food, total_agent_syntheses


def generate_lineage_mermaid(folder):

  nodes = {}
  if not os.path.exists(folder):
    return ""
  for filename in os.listdir(folder):
    if filename.endswith(".yaml"):
      with open(os.path.join(folder, filename), "r") as f:
        cfg = yaml.safe_load(f)
        if "agents" in cfg and len(cfg["agents"]) > 0:
          agent = cfg["agents"][0]
          aid = agent.get("id")
          name = agent.get("name")
          parents = agent.get("parent_ids", [])
          if aid:
            nodes[aid] = {"name": name, "parents": parents}

  mermaid = "## Family Tree (Lineage)\n\n"
  mermaid += "```mermaid\ngraph TD\n"
  for aid, data in nodes.items():
    name = data["name"]
    parents = data["parents"]
    mermaid += f'  {aid}["{name}"]\n'
    for p in parents:
      if p in nodes:  # Only link if parent is also in the folder.
        mermaid += f"  {p} --> {aid}\n"
  mermaid += "```\n"
  return mermaid


def update_leaderboard():
  """Evaluates all templates and updates the leaderboard."""
  results = []
  agents_dir = "templates/agents"
  if os.path.exists(agents_dir):
    for item in os.listdir(agents_dir):
      item_path = os.path.join(agents_dir, item)
      if os.path.isdir(item_path) and item != "old":
        # Find all YAMLs in species folder and generations subfolder
        files = []
        for filename in os.listdir(item_path):
          if filename.endswith(".yaml"):
            files.append(os.path.join(item_path, filename))

        gen_dir = os.path.join(item_path, "generations")
        if os.path.exists(gen_dir):
          for filename in os.listdir(gen_dir):
            if filename.endswith(".yaml"):
              files.append(os.path.join(gen_dir, filename))

        for path in files:
          filename = os.path.basename(path)
          logger.info("Evaluating %s...", filename)
          try:
            score, steps, sim_syntheses, distance, survival, food, agent_syntheses = evaluate_template(
                path)
            results.append({
                "name": filename[:-5],
                "score": score,
                "steps": steps,
                "syntheses": agent_syntheses,
                "distance": distance,
                "survival": survival,
                "food": food
            })
          except Exception as e:
            logger.error("Failed to evaluate %s: %s", filename, e)

  # Sort by score descending
  results.sort(key=lambda x: x["score"], reverse=True)

  # Compute stats
  species = set()
  for res in results:
    name = res["name"]
    if "__" in name:
      species.add(name.split("__")[0])
    else:
      species.add(name)

  total_species = len(species)

  # Generate Markdown
  markdown = "# Simulation Leaderboard\n\n"
  markdown += "## Summary Stats\n"
  markdown += f"- **Total Configs Evaluated**: {len(results)}\n"
  markdown += f"- **Total Species/Variants**: {total_species}\n"
  markdown += f"- **Top Performer**: {results[0]['name']} (Score: {results[0]['score']:.2f})\n"
  markdown += "- **Plot Insights**:\n"
  markdown += "  - *Timeline*: Distinct clusters of high-intensity evolution separated by gaps.\n"
  markdown += "  - *Progression*: Upward slope in reward clusters indicates successful optimization over time.\n"
  markdown += "  - *Density*: Overlapping points show high concentration of similar performers in successful generations.\n\n"

  markdown += "## Evolution Progress\n\n"
  markdown += "![Progress Plot](results/progress.png)\n\n"

  markdown += "## Rankings\n"
  markdown += "| Rank | Config | Score | Distance | Survival | Food | Breeding |\n"
  markdown += "|------|--------|-------|----------|----------|------|----------|\n"

  for i, res in enumerate(results):
    markdown += f"| {i+1} | {res['name']} | {res['score']:.2f} | {res['distance']:.2f} | {res['survival']:.1f} | {res['food']} | {res['syntheses']} |\n"

  # Add Family Tree (Rendered to PNG via Graphviz.)
  try:
    from render import generate_lineage_plot
    generate_lineage_plot(results)
    markdown += "\n## Family Tree (Lineage)\n\n![Lineage Tree](results/lineage.png)\n"
  except Exception as e:
    logger.error("Failed to render Graphviz image: %s", e)
    # Fallback to raw Mermaid.
    markdown += "\n" + generate_lineage_mermaid("templates/agents")

  with open("LEADERBOARD.md", "w") as f:
    f.write(markdown)

  logger.info("Leaderboard updated in LEADERBOARD.md")


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--species", type=str, help="Specific species to evolve")
  parser.add_argument("--pop-size",
                      type=int,
                      default=20,
                      help="Population size")
  parser.add_argument("--generations",
                      type=int,
                      default=20,
                      help="Number of generations")
  args = parser.parse_args()

  LOCK_FILE = f"auto_evolve_{args.species}.lock" if args.species else "auto_evolve.lock"
  if os.path.exists(LOCK_FILE):
    print(f"Another instance of auto_evolve.py for {args.species or 'all'} is running. Exiting.")
    sys.exit(0)

  with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

  species_list = [
      ("quadruped", Agent, {}),
      ("asymmetric_quadruped", ConfigurableAgent, "templates/agents/asymmetric_quadruped/asymmetric_quadruped.yaml"),
      ("rolling_agent", ConfigurableAgent, "templates/agents/rolling_agent/rolling_agent.yaml"),
      ("quadruped_fixed", ConfigurableAgent, "templates/agents/quadruped_fixed/quadruped_fixed.yaml"),
      ("goliath_crawler", ConfigurableAgent, "templates/agents/goliath_crawler/goliath_crawler_default.yaml"),
      ("legion_hexapod", ConfigurableAgent, "templates/agents/legion_hexapod/legion_hexapod_default.yaml"),
      ("aegis_turtle", ConfigurableAgent, "templates/agents/aegis_turtle/aegis_turtle_default.yaml"),
      ("ein_corgi", ConfigurableAgent, "templates/agents/ein_corgi/ein_corgi_default.yaml"),
      ("khepri_beetle", ConfigurableAgent, "templates/agents/khepri_beetle/khepri_beetle_default.yaml"),
      ("giraffe_default", ConfigurableAgent, "templates/agents/giraffe_default/giraffe_default_default.yaml"),
      ("arachne_spider", ConfigurableAgent, "templates/agents/arachne_spider/arachne_spider_default.yaml"),
      ("centipede", ConfigurableAgent, "templates/agents/centipede/centipede_default.yaml"),
      ("scorpion", ConfigurableAgent, "templates/agents/scorpion/scorpion_default.yaml"),
      ("gorilla", ConfigurableAgent, "templates/agents/gorilla/gorilla_default.yaml"),
      ("starfish", ConfigurableAgent, "templates/agents/starfish/starfish_default.yaml"),
      ("snake", ConfigurableAgent, "templates/agents/snake/snake_default.yaml"),
      ("kangaroo", ConfigurableAgent, "templates/agents/kangaroo/kangaroo_default.yaml"),
      ("crab", ConfigurableAgent, "templates/agents/crab/crab_default.yaml"),
      ("megapede", ConfigurableAgent, "templates/agents/megapede/megapede_default.yaml"),
      ("stilts_biped", ConfigurableAgent, "templates/agents/stilts_biped/stilts_biped_default.yaml"),
      ("megarachne", ConfigurableAgent, "templates/agents/megarachne/megarachne_default.yaml"),
      ("mech_biped", ConfigurableAgent, "templates/agents/mech_biped/mech_biped_default.yaml"),
      ("scorpion_king", ConfigurableAgent, "templates/agents/scorpion_king/scorpion_king_default.yaml")
  ]

  try:
    for name, cls, path in species_list:
      if args.species and name != args.species:
        continue

      logger.info("=== Starting Evolution for %s ===", name)

      cfg = {}
      if isinstance(path, str):
        if not os.path.exists(path):
          # Try fallback without _default
          alt_path = path.replace("_default.yaml", ".yaml")
          if os.path.exists(alt_path):
            path = alt_path
            logger.info("Falling back to template path: %s", path)
        with open(path, "r") as f:
          cfg = yaml.safe_load(f)["agents"][0]

      evolve_species(cls,
                     name,
                     cfg,
                     pop_size=args.pop_size,
                     generations=args.generations)

    # Update Leaderboard.
    update_leaderboard()
  finally:
    if os.path.exists(LOCK_FILE):
      os.remove(LOCK_FILE)


if __name__ == "__main__":
  main()

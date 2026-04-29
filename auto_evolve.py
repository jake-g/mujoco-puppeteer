"""Automated evolution system for multiple agent types."""

import logging
import os
import random
from typing import Sequence

import mujoco
import yaml

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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
      results.append((agent, reward))
    except Exception as e:
      logger.error("Failed to get result for %s: %s", agent.name, e)
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

    results_dir = f"results/{agent.name}"
    os.makedirs(results_dir, exist_ok=True)

    steps = int(duration / model.opt.timestep)
    for i in range(steps):
      mujoco.mj_step(model, data)
      if i % 10 == 0:
        renderer.update_scene(data, camera="track_cam")
        pixels = renderer.render()
        frame_filename = f"{results_dir}/frame_{i:05d}.ppm"
        with open(frame_filename, "wb") as f:
          f.write(f"P6\n800 800\n255\n".encode())
          f.write(pixels.tobytes())

    logger.info("Saved %d seconds of frames to %s", int(duration), results_dir)
  except Exception as e:
    logger.error("Failed to save frames for %s: %s", agent.name, e)


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

  display_name = species_name

  population = [
      agent_class(name=f"{display_name}_{i:04x}", config=config)
      for i in range(pop_size)
  ]

  history = []

  for gen in range(generations):
    results = evaluate_population(population, generation=gen)
    logger.info("Gen %d Best Reward: %.2f", gen, results[0][1])

    history.append({"gen": gen, "reward": float(results[0][1])})

    best_agent = results[0][0]

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
          -0.2, 0.2)
      child.phase = child.phase + child.phase * random.uniform(-0.2, 0.2)
      child.amplitude = child.amplitude + child.amplitude * random.uniform(
          -0.2, 0.2)
      child.leg_length_scale = child.leg_length_scale + child.leg_length_scale * random.uniform(
          -0.2, 0.2)
      child.phase_offsets = [
          p + p * random.uniform(-0.2, 0.2) for p in child.phase_offsets
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
  history_path = "results/evolution_history.tsv"
  file_exists = os.path.exists(history_path)

  import time
  with open(history_path, "a") as f:
    if not file_exists:
      f.write("timestamp\tspecies\tgeneration\treward\n")
    for entry in history:
      f.write(
          f"{time.time()}\t{species_name}\t{entry['gen']}\t{entry['reward']}\n")
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

  # Render image for the best agent.
  try:
    env = Environment()
    orch = Orchestrator(env, [best_agent])
    xml_str = orch.generate_combined_xml()

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 400, 400)
    renderer.update_scene(data, camera="main_cam")
    pixels = renderer.render()

    ppm_filename = f"templates/agents/{best_agent.name}.ppm"
    with open(ppm_filename, "wb") as f:
      f.write(f"P6\n400 400\n255\n".encode())
      f.write(pixels.tobytes())

    logger.info("Saved image of agent to %s", ppm_filename)

    save_agent_frames(best_agent, duration=10.0)
  except Exception as e:
    logger.error("Failed to save image or frames for %s: %s", best_agent.name,
                 e)


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

  return avg_reward, total_steps, orchestrator.total_syntheses, avg_distance, avg_survival


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
  folders = ["templates/scenes", "templates/agents"]

  for folder in folders:
    if not os.path.exists(folder):
      continue
    for filename in os.listdir(folder):
      if filename.endswith(".yaml"):
        path = os.path.join(folder, filename)
        logger.info("Evaluating %s...", filename)
      try:
        score, steps, syntheses, distance, survival = evaluate_template(path)
        results.append({
            "name": filename[:-5],
            "score": score,
            "steps": steps,
            "syntheses": syntheses,
            "distance": distance,
            "survival": survival
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
  markdown += f"- **Top Performer**: {results[0]['name']} (Score: {results[0]['score']:.2f})\n\n"

  markdown += "## Evolution Progress\n\n"
  markdown += "![Progress Plot](results/progress.png)\n\n"

  markdown += "## Rankings\n"
  markdown += "| Rank | Config | Score | Steps | Syntheses | Distance | Survival |\n"
  markdown += "|------|--------|-------|-------|-----------|----------|----------|\n"

  for i, res in enumerate(results):
    markdown += f"| {i+1} | {res['name']} | {res['score']:.2f} | {res['steps']} | {res['syntheses']} | {res['distance']:.2f} | {res['survival']:.1f} |\n"

  # Add Family Tree (Rendered to PNG via Graphviz.)
  try:
    import graphviz

    nodes = {}
    folder = "templates/agents"
    if os.path.exists(folder):
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

    # Only keep nodes that have children. (i.e. are parents themselves!)
    nodes_to_keep = set()
    for aid, data in nodes.items():
      for p in data["parents"]:
        if p in nodes:
          nodes_to_keep.add(p)

    dot = graphviz.Digraph(comment='Family Tree')
    dot.attr(dpi='300')  # High definition rendering.
    dot.attr(rankdir='LR')  # Left to Right (makes it less wide.)
    dot.attr(splines='curved')
    dot.attr(nodesep='0.8')
    dot.attr(ranksep='1.5')

    for aid in nodes_to_keep:
      data = nodes[aid]
      name = data["name"]

      # Parse name to make it fit nicely
      parts = name.split("__")
      species = parts[0].replace("_default", "")
      gen = parts[-1] if len(parts) > 1 and "gen" in parts[-1] else ""

      label = f"{species}\n{gen}"

      # Set color based on species
      color = "lightblue"
      if "turtle" in species:
        color = "#a8dadc"
      elif "giraffe" in species:
        color = "#f1faee"
      elif "spider" in species:
        color = "#e63946"
      elif "beetle" in species:
        color = "#f4a261"
      elif "corgi" in species:
        color = "#e9c46a"
      elif "crawler" in species:
        color = "#2a9d8f"

      dot.node(aid, label, style='filled', fillcolor=color, shape='ellipse')

    for aid, data in nodes.items():
      for p in data["parents"]:
        if p in nodes_to_keep and aid in nodes_to_keep:
          # Find child score in results.
          child_score = next(
              (r["score"] for r in results if r["name"] == data["name"]), None)

          label_str = ""
          penwidth = "1.0"
          if child_score is not None:
            label_str = f"S: {child_score:.1f}"
            # Map score to thickness (e.g. score -10 -> 1.0, score 10 -> 5.0)
            penwidth = f"{max(1.0, (child_score + 10.0) / 4.0):.1f}"

          dot.edge(p, aid, xlabel=label_str, penwidth=penwidth)

    dot.render('results/lineage', format='png', cleanup=True)
    markdown += "\n## Family Tree (Lineage)\n\n.[Lineage Tree](results/lineage.png)\n"
    logger.info("Rendered lineage tree to results/lineage.png using Graphviz")
  except Exception as e:
    logger.error("Failed to render Graphviz image: %s", e)
    # Fallback to raw Mermaid.
    markdown += "\n" + generate_lineage_mermaid("templates/agents")

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

  # Evolve Centipede
  with open("templates/agents/centipede_default.yaml", "r") as f:
    centipede_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "centipede",
                 centipede_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Scorpion
  with open("templates/agents/scorpion_default.yaml", "r") as f:
    scorpion_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "scorpion",
                 scorpion_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Gorilla
  with open("templates/agents/gorilla_default.yaml", "r") as f:
    gorilla_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "gorilla",
                 gorilla_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Starfish
  with open("templates/agents/starfish_default.yaml", "r") as f:
    starfish_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "starfish",
                 starfish_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Snake
  with open("templates/agents/snake_default.yaml", "r") as f:
    snake_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "snake",
                 snake_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Kangaroo
  with open("templates/agents/kangaroo_default.yaml", "r") as f:
    kangaroo_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "kangaroo",
                 kangaroo_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Crab
  with open("templates/agents/crab_default.yaml", "r") as f:
    crab_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "crab",
                 crab_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Megapede (Goliath Centipede)
  with open("templates/agents/megapede_default.yaml", "r") as f:
    megapede_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "megapede",
                 megapede_cfg,
                 pop_size=20,
                 generations=20)

  # Evolve Biped (Stilts)
  with open("templates/agents/stilts_biped.yaml", "r") as f:
    biped_cfg = yaml.safe_load(f)["agents"][0]
  evolve_species(ConfigurableAgent,
                 "stilts_biped",
                 biped_cfg,
                 pop_size=20,
                 generations=20)

  # Update Leaderboard.
  update_leaderboard()


if __name__ == "__main__":
  import sys
  LOCK_FILE = "auto_evolve.lock"
  if os.path.exists(LOCK_FILE):
    print("Another instance of auto_evolve.py is running. Exiting.")
    sys.exit(0)

  # Create lock file
  with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

  try:
    main()
  finally:
    # Remove lock file
    if os.path.exists(LOCK_FILE):
      os.remove(LOCK_FILE)

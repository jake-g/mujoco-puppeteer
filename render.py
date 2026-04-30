"""General rendering utilities for MuJoCo simulation."""

import glob
import hashlib
import os
import random
import re

import matplotlib
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import yaml

from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator


def load_yaml(path: str) -> dict:
  """Loads a YAML file."""
  with open(path, "r") as f:
    return yaml.safe_load(f)


def get_species(name: str) -> str:
  """Extracts species name from agent name."""
  if "__" in name:
    return name.split("__")[0]
  species = re.sub(r"_[0-9a-f]+$", "", name)
  species = re.sub(r"_default$", "", species)
  return species


def render_template(template_path: str,
                    output_path: str,
                    output_format: str = "gif",
                    duration: float = 2.0,
                    res: tuple[int, int] = (512, 512),
                    camera: str = "track_cam",
                    zero_gravity: bool = True,
                    floating_spot: bool = True):
  """Renders an agent template to image or GIF."""
  config = load_yaml(template_path)

  env = Environment()
  if output_format == "gif":
    env.floor_size = [0.001, 0.001, 0.001]  # Hide floor for GIFs

  agents = []
  if "agents" in config:
    for agent_cfg in config["agents"]:
      agent = ConfigurableAgent(name=agent_cfg["name"], config=agent_cfg)
      agents.append(agent)

  if not agents:
    print("No agents found in template.")
    return

  orch = Orchestrator(env, agents)
  if output_format == "gif":
    orch.food_positions = []  # No food in GIFs
  orch.initialize()
  assert orch.model is not None
  assert orch.data is not None

  # Use main_cam and position it to look at [0, 0, 1] from distance
  cam_id = mujoco.mj_name2id(orch.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
  if cam_id >= 0:
    # Adaptive offset based on agent size scale
    scale = float(getattr(agents[0], "size_scale", 1.0))
    orch.model.cam_pos[cam_id] = [0.0, -2.5 * scale, 3.5 * scale]

  # Robustly find the free joint of the agent
  jnt_id = -1
  body_id = mujoco.mj_name2id(orch.model, mujoco.mjtObj.mjOBJ_BODY,
                              agents[0].name)
  if body_id >= 0:
    for i in range(orch.model.njnt):
      if orch.model.jnt_bodyid[i] == body_id and orch.model.jnt_type[
          i] == mujoco.mjtJoint.mjJNT_FREE:
        jnt_id = i
        break

  if output_format == "gif":
    frames = []
    renderer = mujoco.Renderer(orch.model, res[0], res[1])

    steps = int(duration / orch.model.opt.timestep)
    for i in range(steps):
      orch.step()

      # Force position to stay at [0, 0, 1] to keep in frame
      if jnt_id >= 0:
        qpos_addr = orch.model.jnt_qposadr[jnt_id]
        orch.data.qpos[qpos_addr:qpos_addr + 3] = [0.0, 0.0, 1.0]
        dof_addr = orch.model.jnt_dofadr[jnt_id]
        orch.data.qvel[dof_addr:dof_addr + 6] = 0.0

      if i % 5 == 0:
        renderer.update_scene(orch.data, camera="main_cam")
        pixels = renderer.render()
        frames.append(Image.fromarray(pixels))

    if frames:
      frames[0].save(output_path,
                     save_all=True,
                     append_images=frames[1:],
                     duration=50,
                     loop=0)
      print(f"Saved GIF to {output_path}")

  elif output_format in ["jpg", "jpeg", "png"]:
    for _ in range(10):
      orch.step()

    renderer = mujoco.Renderer(orch.model, res[0], res[1])

    # Force position to stay at [0, 0, 1] to keep in frame
    if jnt_id >= 0:
      qpos_addr = orch.model.jnt_qposadr[jnt_id]
      orch.data.qpos[qpos_addr:qpos_addr + 3] = [0.0, 0.0, 1.0]
      dof_addr = orch.model.jnt_dofadr[jnt_id]
      orch.data.qvel[dof_addr:dof_addr + 6] = 0.0

    renderer.update_scene(orch.data, camera="main_cam")
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    img.save(output_path,
             output_format.upper() if output_format != "jpg" else "JPEG")
    print(f"Saved {output_format.upper()} to {output_path}")

  elif output_format == "ppm":
    for _ in range(10):
      orch.step()
    renderer = mujoco.Renderer(orch.model, res[0], res[1])
    renderer.update_scene(orch.data, camera="main_cam")
    pixels = renderer.render()

    with open(output_path, "wb") as f:
      f.write(f"P6\n{res[0]} {res[1]}\n255\n".encode())
      f.write(pixels.tobytes())
    print(f"Saved PPM to {output_path}")


def render_scene(template_path: str,
                 output_path: str,
                 res: tuple[int, int] = (1024, 1024)):
  """Renders a scene template to high-res JPEG."""
  config = load_yaml(template_path)

  env = Environment()
  if "environment" in config:
    env_cfg = config["environment"]
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

  agents = []
  if "agents" in config:
    for agent_cfg in config["agents"]:
      agent = ConfigurableAgent(name=agent_cfg["name"], config=agent_cfg)
      agents.append(agent)

  orch = Orchestrator(env, agents)
  orch.initialize()

  for _ in range(10):
    orch.step()
  try:
    for _ in range(10):
      orch.step()

    renderer = mujoco.Renderer(orch.model, res[0], res[1])
    renderer.update_scene(orch.data, camera="main_cam")
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    img.save(output_path, "JPEG")
    print(f"Saved scene render to {output_path}")
  except Exception as e:
    print(f"Failed to render scene {template_path}: {e}")


def create_gif(results_dir="results", species_filter=None, source_dir=None, output_path=None):
  """Generates an evolution GIF for each species in the results directory."""

  if source_dir and output_path:
    print(f"Generating GIF from {source_dir} to {output_path}...")
    frames = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith(".ppm")
    ]
    frames.sort()
    if not frames:
      print(f"No frames found in {source_dir}")
      return

    all_frames = []
    for frame_path in frames:
      try:
        img = Image.open(frame_path)
        all_frames.append(img)
      except Exception as e:
        print(f"Failed to open {frame_path}: {e}")

    if all_frames:
      all_frames[0].save(output_path, save_all=True, append_images=all_frames[1:], duration=50, loop=0)
      print(f"GIF saved successfully to {output_path}!")
    return

  if not os.path.exists(results_dir):
    print(f"Results directory {results_dir} does not exist.")
    return

  agents_dir = os.path.join(results_dir, "agents")
  if not os.path.exists(agents_dir):
    print(f"Agents directory {agents_dir} does not exist.")
    return

  species_dirs = [
      os.path.join(agents_dir, d)
      for d in os.listdir(agents_dir)
      if os.path.isdir(os.path.join(agents_dir, d))
  ]

  species_groups = {}
  for species_dir in species_dirs:
    generations_dir = os.path.join(species_dir, "generations")
    if not os.path.exists(generations_dir):
      continue

    folders = [
        os.path.join(generations_dir, d)
        for d in os.listdir(generations_dir)
        if os.path.isdir(os.path.join(generations_dir, d))
    ]

    species = os.path.basename(species_dir)

    for folder in folders:
      name = os.path.basename(folder)
      gen = 0
      if "__" in name:
        parts = name.split("__")
        try:
          for part in parts:
            if part.startswith("gen"):
              gen = int(part.replace("gen", ""))
              break
        except ValueError:
          pass

      if species not in species_groups:
        species_groups[species] = []
      species_groups[species].append((gen, folder, species_dir))

  from PIL import ImageDraw

  for species, items in species_groups.items():
    if species_filter and species != species_filter:
      continue
    items.sort()

    all_frames = []
    durations = []

    print(f"Processing {len(items)} generations for species: {species}")

    for gen, folder, species_dir in items:
      frames = [
          os.path.join(folder, f)
          for f in os.listdir(folder)
          if f.endswith(".ppm")
      ]
      frames.sort()

      if not frames:
        continue

      step = max(1, len(frames) // 20)
      selected_frames = frames[::step][:20]

      for i, frame_path in enumerate(selected_frames):
        try:
          img = Image.open(frame_path)
          width, height = img.size
          crop_w = width // 2
          crop_h = height // 2
          left = (width - crop_w) // 2
          top = (height - crop_h) // 2
          right = left + crop_w
          bottom = top + crop_h

          img = img.crop((left, top, right, bottom))
          img = img.resize((width, height), Image.LANCZOS)

          draw = ImageDraw.Draw(img)
          text = f"{species} - Gen {gen} ({i+1}/{len(selected_frames)})"
          draw.text((10, 10), text, fill=(255, 255, 255))

          all_frames.append(img)
          durations.append(50)
        except Exception as e:
          print(f"Failed to process {frame_path}: {e}")

    if not all_frames:
      print(f"No frames found for species {species}.")
      continue

    species_dir = items[0][2]
    output_path = os.path.join(species_dir, "evolution.gif")
    print(
        f"Saving GIF for {species} with {len(all_frames)} frames to {output_path}..."
    )
    all_frames[0].save(output_path,
                       save_all=True,
                       append_images=all_frames[1:],
                       duration=durations,
                       loop=0)
    print(f"GIF for {species} saved successfully!")


def rerender_all(agents_dir="templates/agents", use_gif=True):
  """Re-renders all templates in batch mode."""
  if not os.path.exists(agents_dir):
    print(f"Directory {agents_dir} does not exist.")
    return

  subdirs = [
      os.path.join(agents_dir, d)
      for d in os.listdir(agents_dir)
      if os.path.isdir(os.path.join(agents_dir, d)) and d != "old"
  ]

  for subdir in subdirs:
    species = os.path.basename(subdir)
    print(f"Processing species: {species}")

    try:
      files = [f for f in os.listdir(subdir) if f.endswith(".yaml")]
    except FileNotFoundError:
      print(f"Directory {subdir} disappeared. Skipping.")
      continue
    gen_dir = os.path.join(subdir, "generations")
    if os.path.exists(gen_dir):
      files.extend([
          os.path.join("generations", f)
          for f in os.listdir(gen_dir)
          if f.endswith(".yaml")
      ])

    if not files:
      continue

    default_file = None
    best_file = None
    max_gen = -1

    for f in files:
      name, _ = os.path.splitext(os.path.basename(f))
      if name == species or name == f"{species}_default":
        default_file = f

      gen = -1
      parts = name.split("__")
      for part in parts:
        if part.startswith("gen"):
          try:
            gen = int(part.replace("gen", ""))
            break
          except ValueError:
            pass

      if gen > max_gen:
        max_gen = gen
        best_file = f

    if not default_file and files:
      default_file = files[0]

    if default_file:
      gif_path = os.path.join(subdir, f"{species}_default.gif")
      jpg_path = os.path.join(subdir, f"{species}_default.jpg")
      render_template(os.path.join(subdir, default_file),
                      gif_path,
                      output_format="gif")
      render_template(os.path.join(subdir, default_file),
                      jpg_path,
                      output_format="jpg")

    if best_file and best_file != default_file:
      gif_path = os.path.join(subdir, f"{species}_best.gif")
      jpg_path = os.path.join(subdir, f"{species}_best.jpg")
      render_template(os.path.join(subdir, best_file),
                      gif_path,
                      output_format="gif")
      render_template(os.path.join(subdir, best_file),
                      jpg_path,
                      output_format="jpg")


def rerender_scenes(scenes_dir="templates/scenes"):
  """Re-renders all scenes in batch mode."""
  if not os.path.exists(scenes_dir):
    print(f"Directory {scenes_dir} does not exist.")
    return

  files = [f for f in os.listdir(scenes_dir) if f.endswith(".yaml")]
  for f in files:
    path = os.path.join(scenes_dir, f)
    name, _ = os.path.splitext(f)
    output_path = os.path.join(scenes_dir, f"{name}.jpg")
    render_scene(path, output_path)


def generate_plot(results_dir="results"):
  """Generates a progress plot from evolution history files."""
  files = []
  for root, _, fs in os.walk(results_dir):
    for f in fs:
      if f == "evolution_history.tsv":
        files.append(os.path.join(root, f))

  if not files:
    print("No history files found.")
    return

  data = {}
  for f_path in files:
    with open(f_path, "r") as f:
      f.readline()  # Skip header
      for line in f:
        parts = line.strip().split()
        if len(parts) >= 7:
          ts = float(parts[0])
          species = parts[1]
          reward = float(parts[3])
          distance = float(parts[4])
          food = float(parts[5])
          breeding = float(parts[6])

          if species not in data:
            data[species] = []
          data[species].append((ts, reward, distance, food, breeding))
        elif len(parts) >= 4:
          ts = float(parts[0])
          species = parts[1]
          reward = float(parts[3])

          if species not in data:
            data[species] = []
          data[species].append((ts, reward, 0.0, 0.0, 0.0))

  # Find min timestamp for relative X-axis
  # Collect all points to find global gaps
  all_points = []
  for species, points in data.items():
    for pt in points:
      all_points.append((pt[0], species, pt[1]))

  all_points.sort(key=lambda x: x[0])

  # Compress gaps larger than 1 hour
  compressed_data = {}
  if all_points:
    current_time = 0.0
    last_ts = all_points[0][0]
    max_gap = 1.0 # maximum gap in hours to show

    for ts, species, rw in all_points:
      delta = (ts - last_ts) / 3600.0
      if delta > max_gap:
        current_time += max_gap
      else:
        current_time += delta
      last_ts = ts

      if species not in compressed_data:
        compressed_data[species] = []
      compressed_data[species].append((current_time, rw))

  plt.figure(figsize=(12, 8))

  for species, points in compressed_data.items():
    if not points:
      continue
    # Add jitter to X-axis to prevent perfect overlap
    hours = [p[0] + random.uniform(-0.15, 0.15) for p in points]
    rewards = [p[1] for p in points]

    # Try to get color from template
    color = None
    template_path = f"templates/agents/{species}/{species}_default.yaml"
    if os.path.exists(template_path):
      try:
        with open(template_path, "r") as f:
          cfg = yaml.safe_load(f)
          if "agents" in cfg and len(cfg["agents"]) > 0:
            color = cfg["agents"][0].get("plot_color", None)
      except Exception:
        pass

    if color:
      plt.plot(hours, rewards, label=species, marker='o', markersize=3, color=color, linestyle='None', alpha=0.4)
    else:
      plt.plot(hours, rewards, label=species, marker='o', markersize=3, linestyle='None', alpha=0.4)

  plt.xlabel("Time (Hours)")
  plt.ylabel("Best Reward (Fitness)")
  plt.title("Hill Climbing: Species Improvement Over Time")
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True, alpha=0.3)

  plt.tight_layout()
  os.makedirs("results", exist_ok=True)
  plot_path = "results/progress.png"
  plt.savefig(plot_path, dpi=300)
  print(f"Success! Progress plot saved to {plot_path}")
  plt.close()


def generate_lineage_plot(results: list[dict]):
  """Generates a lineage tree plot using Graphviz."""
  import graphviz

  nodes = {}
  score_map = {res["name"]: res["score"] for res in results}

  # Find all YAMLs in templates/agents and subfolders
  for root, _, files in os.walk("templates/agents"):
    if "old" in root:
      continue
    for filename in files:
      if filename.endswith(".yaml"):
        with open(os.path.join(root, filename), "r") as f:
          cfg = yaml.safe_load(f)
          if "agents" in cfg and len(cfg["agents"]) > 0:
            agent = cfg["agents"][0]
            aid = agent.get("id")
            name = agent.get("name")
            parents = agent.get("parent_ids", [])
            if aid:
              nodes[name] = {"aid": aid, "parents": parents, "score": score_map.get(name, 0.0)}



  # Deduplicate by AID to prevent duplicate nodes for same agent
  nodes_by_aid = {}
  for name, data in nodes.items():
    aid = data["aid"]
    if aid not in nodes_by_aid:
      nodes_by_aid[aid] = data
      nodes_by_aid[aid]["name"] = name

  nodes = {data["name"]: data for aid, data in nodes_by_aid.items()}

  # Map ID to name for edge generation
  id_to_name = {data["aid"]: name for name, data in nodes.items()}

  # Only keep nodes that have children
  nodes_to_keep = set()
  for name, data in nodes.items():
    if data["parents"]:
      nodes_to_keep.add(name)
      for p in data["parents"]:
        p_name = id_to_name.get(p)
        if p_name and p_name in nodes:
          nodes_to_keep.add(p_name)

  dot = graphviz.Digraph(comment='Family Tree')
  dot.attr(dpi='300')
  dot.attr(rankdir='LR')
  dot.attr(splines='curved')
  dot.attr(nodesep='0.8')
  dot.attr(ranksep='1.5')

  for name in nodes_to_keep:
    data = nodes[name]
    score = data.get("score", 0.0)
    parts = name.split("__")
    species = parts[0].replace("_default", "")
    gen = parts[-1] if len(parts) > 1 and "gen" in parts[-1] else ""
    label = f"{species}\n{gen}\nS:{score:.1f}"

    h = hashlib.md5(species.encode()).hexdigest()
    color = "#" + h[:6]

    dot.node(name, label, style='filled', fillcolor=color, shape='ellipse')

  for name, data in nodes.items():
    for p in data["parents"]:
      p_name = id_to_name.get(p)
      if p_name in nodes_to_keep and name in nodes_to_keep:
        child_score = next((r["score"] for r in results if r["name"] == name), None)
        label_str = ""
        penwidth = "1.0"
        if child_score is not None:
          label_str = f"S: {child_score:.1f}"
          penwidth = f"{max(1.0, (child_score + 10.0) / 4.0):.1f}"
        dot.edge(p_name, name, xlabel=label_str, penwidth=penwidth)

  dot.render('results/lineage', format='png', cleanup=True)
  print("Success! Lineage tree saved to results/lineage.png")


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("template",
                      type=str,
                      nargs="?",
                      help="Path to template YAML")
  parser.add_argument("output", type=str, nargs="?", help="Output file path")
  parser.add_argument("--format",
                      type=str,
                      default="gif",
                      choices=["gif", "jpg", "jpeg", "png", "ppm"],
                      help="Output format")
  parser.add_argument("--duration",
                      type=float,
                      default=2.0,
                      help="Duration for GIF")
  parser.add_argument("--width", type=int, default=512, help="Image width")
  parser.add_argument("--height", type=int, default=512, help="Image height")
  parser.add_argument("--camera",
                      type=str,
                      default="track_cam",
                      help="Camera name")
  parser.add_argument("--scene",
                      action="store_true",
                      help="Render as scene instead of agent")
  parser.add_argument("--batch",
                      action="store_true",
                      help="Batch render all templates and scenes")

  args = parser.parse_args()

  if args.batch:
    print("Re-rendering agents...")
    rerender_all(use_gif=True)
    print("Re-rendering scenes...")
    rerender_scenes()
    print("All re-rendering finished.")
  else:
    if not args.template or not args.output:
      parser.print_help()
      return

    if args.scene:
      render_scene(args.template, args.output, res=(args.width, args.height))
    else:
      render_template(args.template,
                      args.output,
                      output_format=args.format,
                      duration=args.duration,
                      res=(args.width, args.height),
                      camera=args.camera)


if __name__ == "__main__":
  main()

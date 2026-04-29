import os
import time

from PIL import Image
from PIL import ImageDraw


def create_gif(results_dir="results", species_filter=None):
  """Generates an evolution GIF for each species in the results directory."""
  if not os.path.exists(results_dir):
    print(f"Results directory {results_dir} does not exist.")
    return

  folders = [
      os.path.join(results_dir, d)
      for d in os.listdir(results_dir)
      if os.path.isdir(os.path.join(results_dir, d))
  ]

  # Group folders by species
  species_groups = {}
  for folder in folders:
    name = os.path.basename(folder)
    if "__" in name:
      parts = name.split("__")
      species = parts[0]
      gen = 0
      try:
        # Extract generation number
        for part in parts:
          if part.startswith("gen"):
            gen = int(part.replace("gen", ""))
            break
      except ValueError:
        pass

      if species not in species_groups:
        species_groups[species] = []
      species_groups[species].append((gen, folder))

  # For each species, create a GIF
  for species, items in species_groups.items():
    if species_filter and species != species_filter:
      continue
    # Sort by generation to show progression
    items.sort()

    all_frames = []
    durations = []

    print(f"Processing {len(items)} generations for species: {species}")

    for gen, folder in items:
      frames = [
          os.path.join(folder, f)
          for f in os.listdir(folder)
          if f.endswith(".ppm")
      ]
      frames.sort()

      if not frames:
        continue

      # Take up to 20 frames per folder to prevent OOM crashes.
      step = max(1, len(frames) // 20)
      selected_frames = frames[::step][:20]

      for i, frame_path in enumerate(selected_frames):
        try:
          img = Image.open(frame_path)
          draw = ImageDraw.Draw(img)

          # Add text overlay with larger font.
          # Try to load Arial font, fallback to default
          try:
            from PIL import ImageFont
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 24)
          except IOError:
            font = ImageFont.load_default()

          text = f"{species} - Gen {gen} ({i+1}/{len(selected_frames)})"
          draw.text((10, 10), text, fill=(255, 255, 255), font=font)

          all_frames.append(img)
          # 50ms per frame (20 fps) to make it less rapid and easier to follow.
          durations.append(50)
        except Exception as e:
          print(f"Failed to process {frame_path}: {e}")

    if not all_frames:
      print(f"No frames found for species {species}.")
      continue

    output_path = os.path.join(results_dir, f"{species}_evolution.gif")
    print(
        f"Saving GIF for {species} with {len(all_frames)} frames to {output_path}..."
    )
    all_frames[0].save(output_path,
                       save_all=True,
                       append_images=all_frames[1:],
                       duration=durations,
                       loop=0)
    print(f"GIF for {species} saved successfully!")


if __name__ == "__main__":
  create_gif()

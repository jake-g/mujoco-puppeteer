"""Environment module for MuJoCo simulation."""

import logging
import xml.etree.ElementTree as ET

import mujoco

logger = logging.getLogger(__name__)


class Environment:
  """Manages the MuJoCo simulation environment (world and physics)."""

  def __init__(self):
    """Initializes the environment with default settings."""
    self.gravity = [0.0, 0.0, -9.81]
    self.wind = [0.0, 0.0, 0.0]  # Default no wind
    self.floor_size = [10.0, 10.0, 0.05]
    self.floor_rgb1 = [0.0, 0.0, 0.0]  # Black
    self.floor_rgb2 = [1.0, 1.0, 1.0]  # White
    self.sky_rgb1 = [0.6, 0.8, 1.0]  # Light Blue
    self.sky_rgb2 = [1.0, 1.0, 1.0]  # White
    self.obstacles = []
    self.rough_terrain = False  # Procedural rough terrain
    self.timestep = 0.005  # Default timestep for faster simulation
    self.camera_pos = [0.0, -20.0, 20.0]
    self.camera_xyaxes = [1.0, 0.0, 0.0, 0.0, 0.707, 0.707]
    self.camera_distance = 28.28
    self.camera_elevation = -45.0
    self.camera_azimuth = 90.0

  def generate_xml(self) -> str:
    """Generates the MJCF XML string for the environment.

    Returns:
        str: The XML string.
    """
    root = ET.Element("mujoco", model="grid_world")

    # Option
    option = ET.SubElement(root, "option")
    option.set("gravity", " ".join(map(str, self.gravity)))
    option.set("wind", " ".join(map(str, self.wind)))
    if hasattr(self, 'timestep') and self.timestep is not None:
      option.set("timestep", str(self.timestep))

    # Visual settings for offscreen rendering (prevent framebuffer size errors)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1024", offheight="1024")

    # Asset
    asset = ET.SubElement(root, "asset")

    # Skybox
    skybox = ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        width="256",
        height="256",
    )
    skybox.set("rgb1", " ".join(map(str, self.sky_rgb1)))
    skybox.set("rgb2", " ".join(map(str, self.sky_rgb2)))

    texture = ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
    )
    texture.set(
        "rgb1",
        f"{self.floor_rgb1[0]} {self.floor_rgb1[1]} {self.floor_rgb1[2]}")
    texture.set(
        "rgb2",
        f"{self.floor_rgb2[0]} {self.floor_rgb2[1]} {self.floor_rgb2[2]}")

    material = ET.SubElement(asset,
                             "material",
                             name="grid",
                             texture="grid",
                             texrepeat="1 1")
    material.set("texuniform", "true")
    material.set("reflectance", "0.2")

    # Worldbody
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody,
                  "light",
                  diffuse=".5 .5 .5",
                  pos="0 0 3",
                  dir="0 0 -1")

    # Add a default camera that is zoomed out for better viewing
    ET.SubElement(
        worldbody,
        "camera",
        name="main_cam",
        pos=f"{self.camera_pos[0]} {self.camera_pos[1]} {self.camera_pos[2]}",
        xyaxes=
        f"{self.camera_xyaxes[0]} {self.camera_xyaxes[1]} {self.camera_xyaxes[2]} {self.camera_xyaxes[3]} {self.camera_xyaxes[4]} {self.camera_xyaxes[5]}"
    )

    floor = ET.SubElement(worldbody, "geom", name="floor", type="plane")
    floor.set(
        "size",
        f"{self.floor_size[0]} {self.floor_size[1]} {self.floor_size[2]}")
    floor.set("material", "grid")
    floor.set("condim", "3")
    floor.set("friction",
              "5.0 0.005 0.0001")  # Heavily increased friction for grip
    floor.set("solref", "0.002 1")  # Make floor stiffer to prevent sinking
    floor.set("solimp", "0.9 0.95 0.001")

    # Add rough terrain if enabled.
    if self.rough_terrain:
      import random as py_random
      spacing = 0.5
      grid_size = int(self.floor_size[0] / spacing)
      for x in range(-grid_size, grid_size):
        for y in range(-grid_size, grid_size):
          # Skip center to let agents spawn safely.
          if abs(x) < 4 and abs(y) < 4:
            continue
          height = py_random.uniform(0.01, 0.04)
          ET.SubElement(
              worldbody,
              "geom",
              type="box",
              size=f"0.2 0.2 {height}",
              pos=f"{x * spacing} {y * spacing} {height}",
              rgba="0.8 0.8 0.8 1",
          )

    # Add obstacles
    for i, obs in enumerate(self.obstacles):
      obs_body = ET.SubElement(
          worldbody,
          "body",
          name=f"obstacle_{i}",
          pos=f"{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}",
      )
      ET.SubElement(obs_body, "freejoint", name=f"obstacle_{i}_joint")
      ET.SubElement(
          obs_body,
          "geom",
          type=obs.get("type", "box"),
          size=f"{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}",
          mass=str(obs.get("mass", 1.0)),
          rgba=
          f"{obs.get('color', [0.5, 0.5, 0.5, 1.0])[0]} {obs.get('color', [0.5, 0.5, 0.5, 1.0])[1]} {obs.get('color', [0.5, 0.5, 0.5, 1.0])[2]} {obs.get('color', [0.5, 0.5, 0.5, 1.0])[3]}",
      )

    return ET.tostring(root, encoding="unicode")

  def create_model(self) -> mujoco.MjModel:
    """Creates the MuJoCo model from the generated XML.

    Returns:
        mujoco.MjModel: The compiled model.
    """
    xml_str = self.generate_xml()
    logger.debug("Generated XML: %s", xml_str)
    return mujoco.MjModel.from_xml_string(xml_str)

  def set_gravity(self, gravity: list[float]):
    """Sets the gravity vector.

    Args:
        gravity: A list of 3 floats representing the gravity vector.
    """
    if len(gravity) != 3:
      raise ValueError("Gravity must be a list of 3 floats.")
    self.gravity = gravity

  def update_runtime_physics(self, model: mujoco.MjModel, data: mujoco.MjData):
    """Updates physics parameters at runtime.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
    """
    # In MuJoCo, gravity is stored in model.opt.gravity
    if list(model.opt.gravity) != self.gravity:
      model.opt.gravity[:] = self.gravity
      logger.info("Updated runtime gravity to: %s", self.gravity)

    # Update wind
    if list(model.opt.wind) != self.wind:
      model.opt.wind[:] = self.wind
      logger.info("Updated runtime wind to: %s", self.wind)

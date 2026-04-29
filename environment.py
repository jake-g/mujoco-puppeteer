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
    self.floor_size = [10.0, 10.0, 0.05]
    self.floor_rgb1 = [0.0, 0.0, 0.0]  # Black
    self.floor_rgb2 = [1.0, 1.0, 1.0]  # White
    self.sky_rgb1 = [0.6, 0.8, 1.0]  # Light Blue
    self.sky_rgb2 = [1.0, 1.0, 1.0]  # White

  def generate_xml(self) -> str:
    """Generates the MJCF XML string for the environment.

    Returns:
        str: The XML string.
    """
    root = ET.Element("mujoco", model="grid_world")

    # Option
    option = ET.SubElement(root, "option")
    option.set("gravity",
               f"{self.gravity[0]} {self.gravity[1]} {self.gravity[2]}")

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
    skybox.set("rgb1",
               f"{self.sky_rgb1[0]} {self.sky_rgb1[1]} {self.sky_rgb1[2]}")
    skybox.set("rgb2",
               f"{self.sky_rgb2[0]} {self.sky_rgb2[1]} {self.sky_rgb2[2]}")

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

    # Add a default camera that is zoomed out
    ET.SubElement(worldbody, "camera", name="main_cam", pos="0 -15 15")

    floor = ET.SubElement(worldbody, "geom", name="floor", type="plane")
    floor.set(
        "size",
        f"{self.floor_size[0]} {self.floor_size[1]} {self.floor_size[2]}")
    floor.set("material", "grid")
    floor.set("condim", "3")
    floor.set("friction", "1 0.005 0.0001")  # Increased friction

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

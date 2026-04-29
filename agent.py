"""Agent module for MuJoCo simulation."""

import hashlib
import logging
import math
import random
from typing import Optional
import xml.etree.ElementTree as ET

import mujoco

logger = logging.getLogger(__name__)


class Agent:
  """Represents an agent in the MuJoCo simulation."""

  def __init__(self,
               name: str = "agent_0",
               size_scale: float = 1.0,
               config: Optional[dict] = None):
    """Initializes the agent.

    Args:
        name: Unique name for the agent.
        size_scale: Scaling factor for the agent's size.
    """
    self.name = name
    self.pos = [0.0, 0.0, 1.0]
    self.reward = 0.0
    # Default to a random bright color
    self.color = [random.random(), random.random(), random.random(), 1.0]
    self.cooldown = 0.0
    self.size_scale = size_scale
    self.fallen_time = 0.0
    # Evolution parameters for sine wave policy
    self.frequency = random.uniform(2.0, 5.0)
    self.phase = random.uniform(0.0, 2 * math.pi)
    self.phase_offsets = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    self.amplitude = random.uniform(0.5, 2.0)
    self.steps: int = 0
    self.last_foot_touch: Optional[str] = None
    self.leg_length_scale: float = random.uniform(0.5, 1.5)
    self.time_alive: float = 0.0
    self.saved: bool = False

    # Generate unique ID based on genome hash
    genome = [
        self.frequency, self.phase, self.amplitude, self.leg_length_scale
    ] + self.phase_offsets
    self.id: str = hashlib.md5(str(genome).encode()).hexdigest()[:8]

  def update_id(self):
    """Updates the unique ID based on current genome."""
    genome = [
        self.frequency, self.phase, self.amplitude, self.leg_length_scale
    ] + self.phase_offsets
    self.id = hashlib.md5(str(genome).encode()).hexdigest()[:8]

  def generate_xml(self) -> ET.Element:
    """Generates the XML element for the agent.

    Returns:
        ET.Element: The XML element representing the agent body.
    """
    body = ET.Element("body",
                      name=self.name,
                      pos=f"{self.pos[0]} {self.pos[1]} {self.pos[2]}")
    ET.SubElement(body, "freejoint", name=self.name)

    # Main body
    s = self.size_scale
    ET.SubElement(
        body,
        "geom",
        name=f"{self.name}_geom",
        type="box",
        size=f"{0.1 * s} {0.1 * s} {0.2 * s}",
        rgba=f"{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}",
    )

    # Left leg
    left_leg = ET.SubElement(body,
                             "body",
                             name=f"{self.name}_left_leg",
                             pos=f"0 {0.1 * s} {-0.2 * s}")
    ET.SubElement(
        left_leg,
        "joint",
        name=f"{self.name}_left_hip",
        type="hinge",
        axis="0 1 0",
        range="-45 45",
        limited="true",
    )
    ET.SubElement(left_leg,
                  "geom",
                  type="capsule",
                  size=f"{0.03 * s} {0.1 * s * self.leg_length_scale}",
                  pos=f"0 0 {-0.1 * s * self.leg_length_scale}")

    # Right leg
    right_leg = ET.SubElement(body,
                              "body",
                              name=f"{self.name}_right_leg",
                              pos=f"0 {-0.1 * s} {-0.2 * s}")
    ET.SubElement(
        right_leg,
        "joint",
        name=f"{self.name}_right_hip",
        type="hinge",
        axis="0 1 0",
        range="-45 45",
        limited="true",
    )
    ET.SubElement(right_leg,
                  "geom",
                  type="capsule",
                  size=f"{0.03 * s} {0.1 * s * self.leg_length_scale}",
                  pos=f"0 0 {-0.1 * s * self.leg_length_scale}")

    return body

  def generate_actuators_xml(self) -> list[ET.Element]:
    """Generates the XML elements for the agent's actuators.

    Returns:
        list[ET.Element]: A list of XML elements representing actuators.
    """
    actuators = []

    left_motor = ET.Element(
        "motor",
        name=f"{self.name}_left_motor",
        joint=f"{self.name}_left_hip",
        ctrlrange="-1 1",
    )
    actuators.append(left_motor)

    right_motor = ET.Element(
        "motor",
        name=f"{self.name}_right_motor",
        joint=f"{self.name}_right_hip",
        ctrlrange="-1 1",
    )
    actuators.append(right_motor)

    return actuators

  def calculate_reward(self, data: mujoco.MjData) -> float:
    """Calculates the reward for the agent.

    The default reward is for walking forward (positive x) without falling.

    Args:
        data: The MuJoCo data containing current state.

    Returns:
        float: The calculated reward.
    """
    try:
      agent_body = data.body(self.name)
      pos = agent_body.xpos

      # Reward for moving forward in x direction
      x_reward = pos[0]

      # Penalty for falling (z coordinate dropping below a threshold)
      z_penalty = 0.0
      if pos[2] < 0.5:
        z_penalty = -10.0

      # Survival reward for staying upright
      survival_reward = 0.0
      if pos[2] >= 0.5:
        survival_reward = 0.1

      self.reward = x_reward + z_penalty + survival_reward
      return self.reward
    except Exception as e:
      logger.error("Failed to calculate reward for %s: %s", self.name, e)
      return 0.0

  def act(self, data: mujoco.MjData):
    """Applies controls to actuators based on internal policy."""
    try:
      # Find actuator indices
      left_motor_name = f"{self.name}_left_motor"
      right_motor_name = f"{self.name}_right_motor"

      left_motor_idx = mujoco.mj_name2id(data.model,
                                         mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         left_motor_name)
      right_motor_idx = mujoco.mj_name2id(data.model,
                                          mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          right_motor_name)

      if left_motor_idx >= 0 and right_motor_idx >= 0:
        t = data.time
        # Apply sine wave control
        data.ctrl[left_motor_idx] = math.sin(t * self.frequency + self.phase)
        data.ctrl[right_motor_idx] = math.cos(t * self.frequency + self.phase)
    except Exception as e:
      logger.error("Failed to act for %s: %s", self.name, e)


class QuadrupedAgent(Agent):
  """A 4-legged agent variant for better stability."""

  def generate_xml(self) -> ET.Element:
    """Generates the XML element for the quadruped agent.

    Returns:
        ET.Element: The XML element representing the agent body.
    """
    s = self.size_scale
    body = ET.Element("body",
                      name=self.name,
                      pos=f"{self.pos[0]} {self.pos[1]} {self.pos[2]}")
    ET.SubElement(body, "freejoint", name=self.name)

    # Main body (flatter and wider for stability)
    ET.SubElement(
        body,
        "geom",
        name=f"{self.name}_geom",
        type="box",
        size=f"{0.2 * s} {0.2 * s} {0.05 * s}",
        rgba=f"{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}",
    )

    # 4 Legs at corners
    legs = [
        ("front_left", [0.15 * s, 0.15 * s]),
        ("front_right", [0.15 * s, -0.15 * s]),
        ("back_left", [-0.15 * s, 0.15 * s]),
        ("back_right", [-0.15 * s, -0.15 * s]),
    ]

    for leg_name, pos in legs:
      leg_body = ET.SubElement(body,
                               "body",
                               name=f"{self.name}_{leg_name}_leg",
                               pos=f"{pos[0]} {pos[1]} 0")
      ET.SubElement(
          leg_body,
          "joint",
          name=f"{self.name}_{leg_name}_hip",
          type="hinge",
          axis="0 1 0" if "front" in leg_name else "1 0 0",
          range="-45 45",
          limited="true",
      )
      ET.SubElement(
          leg_body,
          "geom",
          name=f"{self.name}_{leg_name}_geom",
          type="capsule",
          size=f"{0.02 * s} {0.1 * s * self.leg_length_scale}",
          pos=f"0 0 {-0.05 * s * self.leg_length_scale}",
      )

    return body

  def generate_actuators_xml(self) -> list[ET.Element]:
    """Generates the XML elements for the agent's actuators."""
    actuators = []
    legs = ["front_left", "front_right", "back_left", "back_right"]

    for leg_name in legs:
      motor = ET.Element(
          "motor",
          name=f"{self.name}_{leg_name}_motor",
          joint=f"{self.name}_{leg_name}_hip",
          ctrlrange="-1 1",
      )
      actuators.append(motor)

    return actuators

  def act(self, data: mujoco.MjData):
    """Applies controls to all 4 actuators."""
    try:
      legs = ["front_left", "front_right", "back_left", "back_right"]
      t = data.time

      for i, leg_name in enumerate(legs):
        motor_name = f"{self.name}_{leg_name}_motor"
        motor_idx = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      motor_name)

        if motor_idx >= 0:
          # Offset phase for each leg to create a walking gait
          phase_offset = self.phase_offsets[i]
          data.ctrl[motor_idx] = self.amplitude * math.sin(t * self.frequency +
                                                           self.phase +
                                                           phase_offset)
    except Exception as e:
      logger.error("Failed to act for %s: %s", self.name, e)

  def calculate_reward(self, data: mujoco.MjData) -> float:
    """Calculates reward based on distance and steps made."""
    try:
      body = data.body(self.name)
      pos = body.xpos

      # Reward for moving forward
      x_reward = pos[0]

      # Penalty for falling
      z_penalty = 0.0
      if pos[2] < 0.3:  # Lower threshold for quadruped
        z_penalty = -10.0

      # Step detection
      legs = ["front_left", "front_right", "back_left", "back_right"]
      step_bonus = 0.0

      for leg_name in legs:
        try:
          geom_name = f"{self.name}_{leg_name}_geom"
          geom_handle = data.geom(geom_name)
          geom_pos = geom_handle.xpos

          if geom_pos[2] < 0.05:  # Touching ground
            if self.last_foot_touch != leg_name:
              self.steps += 1
              self.last_foot_touch = leg_name
              logger.info("%s made a step with %s", self.name, leg_name)
              step_bonus += 10.0  # High reward for step!
        except Exception:
          pass

      # Survival reward for staying upright
      survival_reward = 0.0
      if pos[2] >= 0.3:
        survival_reward = 0.1

      self.reward = x_reward + z_penalty + step_bonus + survival_reward
      return self.reward
    except Exception as e:
      logger.error("Failed to calculate reward for %s: %s", self.name, e)
      return 0.0


class ConfigurableAgent(Agent):
  """An agent that can be configured abstractly via YAML properties."""

  def __init__(self,
               name: str = "agent_0",
               size_scale: float = 1.0,
               config: Optional[dict] = None):
    super().__init__(name, size_scale)
    self.config = config or {}

    # Parse limbs to determine number of phase offsets needed!
    self.limbs = self.config.get("limbs", [])
    if self.limbs:
      total_actuators = sum(2 if "child" in l else 1 for l in self.limbs)
      self.phase_offsets = [0.0] * total_actuators  # Default to synchronized

  def generate_xml(self) -> ET.Element:
    """Generates the XML element for the configurable agent.

    Returns:
        ET.Element: The XML element representing the agent body.
    """
    s = self.size_scale
    body_cfg = self.config.get("body", {})

    body = ET.Element("body",
                      name=self.name,
                      pos=f"{self.pos[0]} {self.pos[1]} {self.pos[2]}")
    ET.SubElement(body, "freejoint", name=self.name)

    # Main body
    body_type = body_cfg.get("type", "box")
    body_size = body_cfg.get("size", [0.2, 0.2, 0.05])
    body_size_str = " ".join(str(v * s) for v in body_size)

    geom_attrib = {
        "name":
            f"{self.name}_geom",
        "type":
            body_type,
        "size":
            body_size_str,
        "rgba":
            f"{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}",
    }
    if "mass" in body_cfg:
      geom_attrib["mass"] = str(body_cfg["mass"] * s)

    ET.SubElement(body, "geom", attrib=geom_attrib)

    # Limbs
    for limb in self.limbs:
      limb_name = limb["name"]
      limb_pos = limb.get("pos", [0, 0, 0])
      limb_pos_str = " ".join(str(v * s) for v in limb_pos)

      limb_body = ET.SubElement(body,
                                "body",
                                name=f"{self.name}_{limb_name}_leg",
                                pos=limb_pos_str)

      # Hip Joint
      axis = limb.get("axis", [0, 1, 0])
      axis_str = " ".join(str(v) for v in axis)
      joint_range = limb.get("range", [-30, 30])
      range_str = f"{joint_range[0]} {joint_range[1]}"

      ET.SubElement(
          limb_body,
          "joint",
          name=f"{self.name}_{limb_name}_hip",
          type="hinge",
          axis=axis_str,
          range=range_str,
          limited="true",
      )

      # Hip Geom
      geom_cfg = limb.get("geom", {"type": "capsule", "size": [0.02, 0.1]})
      geom_type = geom_cfg.get("type", "capsule")
      geom_size = geom_cfg.get("size", [0.02, 0.1])
      geom_size_str = " ".join(str(v * s) for v in geom_size)

      g_pos = "0 0 0"
      if geom_type == "capsule" and len(geom_size) >= 2:
        g_pos = f"0 0 {-geom_size[1] * s}"

      ET.SubElement(limb_body,
                    "geom",
                    name=f"{self.name}_{limb_name}_geom",
                    type=geom_type,
                    size=geom_size_str,
                    pos=g_pos)

      # Child Limb (Optional, e.g. Knee/Calf)
      if "child" in limb:
        child = limb["child"]
        child_name = child["name"]

        # Position child at the end of parent capsule!
        c_pos = [0.0, 0.0, 0.0]
        if geom_type == "capsule" and len(geom_size) >= 2:
          c_pos = [0.0, 0.0, -2 * geom_size[1] * s]

        c_pos_str = " ".join(str(v) for v in c_pos)

        child_body = ET.SubElement(limb_body,
                                   "body",
                                   name=f"{self.name}_{child_name}_calf",
                                   pos=c_pos_str)

        # Knee Joint
        c_axis = child.get("axis", [0, 1, 0])
        c_axis_str = " ".join(str(v) for v in c_axis)
        c_range = child.get("range", [0, 90])
        c_range_str = f"{c_range[0]} {c_range[1]}"

        ET.SubElement(
            child_body,
            "joint",
            name=f"{self.name}_{child_name}_knee",
            type="hinge",
            axis=c_axis_str,
            range=c_range_str,
            limited="true",
        )

        # Knee Geom
        c_geom_cfg = child.get("geom", {
            "type": "capsule",
            "size": [0.015, 0.1]
        })
        c_geom_type = c_geom_cfg.get("type", "capsule")
        c_geom_size = c_geom_cfg.get("size", [0.015, 0.1])
        c_geom_size_str = " ".join(str(v * s) for v in c_geom_size)

        cg_pos = "0 0 0"
        if c_geom_type == "capsule" and len(c_geom_size) >= 2:
          cg_pos = f"0 0 {-c_geom_size[1] * s}"

        ET.SubElement(child_body,
                      "geom",
                      name=f"{self.name}_{child_name}_geom",
                      type=c_geom_type,
                      size=c_geom_size_str,
                      pos=cg_pos)

    return body

  def generate_actuators_xml(self) -> list[ET.Element]:
    """Generates the XML elements for the agent's actuators."""
    actuators = []
    for limb in self.limbs:
      limb_name = limb["name"]
      motor = ET.Element(
          "motor",
          name=f"{self.name}_{limb_name}_motor",
          joint=f"{self.name}_{limb_name}_hip",
          ctrlrange="-1 1",
      )
      actuators.append(motor)

      if "child" in limb:
        child_name = limb["child"]["name"]
        c_motor = ET.Element(
            "motor",
            name=f"{self.name}_{child_name}_motor",
            joint=f"{self.name}_{child_name}_knee",
            ctrlrange="-1 1",
        )
        actuators.append(c_motor)

    return actuators

  def act(self, data: mujoco.MjData):
    """Applies controls to all actuators."""
    try:
      t = data.time
      motor_idx = 0
      for limb in self.limbs:
        limb_name = limb["name"]
        m_name = f"{self.name}_{limb_name}_motor"
        m_idx = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  m_name)

        if m_idx >= 0:
          phase_offset = self.phase_offsets[motor_idx]
          data.ctrl[m_idx] = self.amplitude * math.sin(t * self.frequency +
                                                       self.phase +
                                                       phase_offset)
          motor_idx += 1

        if "child" in limb:
          child_name = limb["child"]["name"]
          c_m_name = f"{self.name}_{child_name}_motor"
          c_m_idx = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      c_m_name)

          if c_m_idx >= 0:
            phase_offset = self.phase_offsets[motor_idx]
            data.ctrl[c_m_idx] = self.amplitude * math.sin(t * self.frequency +
                                                           self.phase +
                                                           phase_offset)
            motor_idx += 1

    except Exception as e:
      logger.error("Failed to act for %s: %s", self.name, e)

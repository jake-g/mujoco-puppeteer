"""Agent module for MuJoCo simulation."""

import logging
import math
import random
import xml.etree.ElementTree as ET

import mujoco

logger = logging.getLogger(__name__)


class Agent:
  """Represents an agent in the MuJoCo simulation."""

  def __init__(self, name: str = "agent_0", size_scale: float = 1.0):
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
    )
    ET.SubElement(left_leg,
                  "geom",
                  type="capsule",
                  size=f"{0.03 * s} {0.1 * s}",
                  pos=f"0 0 {-0.1 * s}")

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
    )
    ET.SubElement(right_leg,
                  "geom",
                  type="capsule",
                  size=f"{0.03 * s} {0.1 * s}",
                  pos=f"0 0 {-0.1 * s}")

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

      self.reward = x_reward + z_penalty
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
      )
      ET.SubElement(
          leg_body,
          "geom",
          type="capsule",
          size=f"{0.02 * s} {0.1 * s}",
          pos=f"0 0 {-0.05 * s}",
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
          data.ctrl[motor_idx] = math.sin(t * self.frequency + self.phase +
                                          phase_offset)
    except Exception as e:
      logger.error("Failed to act for %s: %s", self.name, e)

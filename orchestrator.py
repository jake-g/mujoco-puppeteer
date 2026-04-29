"""Orchestrator module for MuJoCo simulation."""

import logging
import os
import random
from typing import Sequence
import xml.etree.ElementTree as ET

import mujoco
import yaml

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment

logger = logging.getLogger(__name__)


class Orchestrator:
  """Orchestrates the simulation, environment, and agents."""

  def __init__(self,
               env: Environment,
               agents: Sequence[Agent],
               death_threshold: float = 3.0):
    """Initializes the orchestrator.

    Args:
        env: The environment instance.
        agents: A list of agent instances.
        death_threshold: Time in seconds before a fallen agent dies.
    """
    self.env = env
    self.agents = agents
    self.model = None
    self.data = None
    self.rewards: dict[str, float] = {}
    self.death_threshold = death_threshold

  def generate_combined_xml(self) -> str:
    """Generates the combined MJCF XML string.

    Returns:
        str: The XML string.
    """
    # Start with environment XML
    env_xml = self.env.generate_xml()
    root = ET.fromstring(env_xml)

    worldbody = root.find("worldbody")
    if worldbody is None:
      worldbody = ET.SubElement(root, "worldbody")

    # Add agents to worldbody
    for agent in self.agents:
      worldbody.append(agent.generate_xml())

    # Add actuators
    actuator = root.find("actuator")
    if actuator is None:
      actuator = ET.SubElement(root, "actuator")

    for agent in self.agents:
      for act in agent.generate_actuators_xml():
        act_name = act.get("name")
        logger.info("Adding actuator: %s", act_name)
        actuator.append(act)

    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str

  def initialize(self):
    """Initializes the MuJoCo model and data."""
    xml_str = self.generate_combined_xml()
    logger.debug("Combined XML: %s", xml_str)
    self.model = mujoco.MjModel.from_xml_string(xml_str)
    self.data = mujoco.MjData(self.model)
    # Compute initial positions
    mujoco.mj_forward(self.model, self.data)
    self.rewards = {agent.name: 0.0 for agent in self.agents}

  def step(self):
    """Advances the simulation by one step and updates rewards."""
    if self.model is None or self.data is None:
      raise RuntimeError(
          "Orchestrator not initialized. Call initialize() first.")

    # Apply controls based on agent policy
    for agent in self.agents:
      agent.act(self.data)

    mujoco.mj_step(self.model, self.data)

    # Update rewards
    for agent in self.agents:
      self.rewards[agent.name] = agent.calculate_reward(self.data)

    # Check for death on fall and update time_alive
    for agent in self.agents:
      try:
        body = self.data.body(agent.name)
        agent.time_alive += self.model.opt.timestep

        # Auto-save if survived 30 seconds and not saved yet!
        if agent.time_alive > 30.0 and not agent.saved:
          self._save_agent_config(agent)
          agent.saved = True

        if body.xpos[2] < 0.5:
          agent.fallen_time += self.model.opt.timestep
          if agent.fallen_time > self.death_threshold:
            logger.info("Agent %s died after falling for %.1f seconds",
                        agent.name, self.death_threshold)
            self._respawn_agent(agent)
        else:
          agent.fallen_time = 0.0
      except Exception as e:
        logger.error("Failed to check death for %s: %s", agent.name, e)

    # Check for synthesis
    self._check_synthesis()

  def _check_synthesis(self):
    """Checks for collisions between agents and handles synthesis."""
    if self.data is None or self.model is None:
      return

    # Update cooldowns
    for agent in self.agents:
      if agent.cooldown > 0:
        agent.cooldown -= self.model.opt.timestep

    # Check contacts
    for i in range(self.data.ncon):
      contact = self.data.contact[i]
      geom1_id = contact.geom1
      geom2_id = contact.geom2

      # Get geom names
      geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                     geom1_id)
      geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                     geom2_id)

      if not geom1_name or not geom2_name:
        continue

      # Check if both are agents
      agent1 = None
      agent2 = None
      for agent in self.agents:
        if geom1_name.startswith(agent.name):
          agent1 = agent
        if geom2_name.startswith(agent.name):
          agent2 = agent

      if agent1 and agent2 and agent1 != agent2:
        # Collision between different agents!
        if agent1.cooldown <= 0 and agent2.cooldown <= 0:
          logger.info("Synthesis triggered between %s and %s", agent1.name,
                      agent2.name)
          self._synthesize_agents(agent1, agent2)
          break  # Only synthesize once per step

  def _synthesize_agents(self, parent1: Agent, parent2: Agent):
    """Creates a new agent from two parents."""
    assert self.data is not None
    # Mix colors
    new_color = [(c1 + c2) / 2 for c1, c2 in zip(parent1.color, parent2.color)]

    # Mutate color slightly
    mutation_rate = 0.1
    new_color = [
        min(1.0, max(0.0, c + random.uniform(-mutation_rate, mutation_rate)))
        for c in new_color[:3]
    ] + [1.0]  # Keep alpha=1.0

    # Determine species name for naming
    p1_type = parent1.config.get("type", "Agent") if isinstance(
        parent1, ConfigurableAgent) else parent1.__class__.__name__
    p2_type = parent2.config.get("type", "Agent") if isinstance(
        parent2, ConfigurableAgent) else parent2.__class__.__name__

    if p1_type == p2_type:
      species_prefix = p1_type.capitalize()
    else:
      species_prefix = f"{p1_type.capitalize()}_{p2_type.capitalize()}_Hybrid"

    new_agent = Agent(name="temp_agent")

    new_agent.color = new_color
    new_agent.parent_ids = [parent1.id, parent2.id]

    # Inherit and mutate size
    new_agent.size_scale = (parent1.size_scale +
                            parent2.size_scale) / 2 + random.uniform(-0.1, 0.1)
    new_agent.size_scale = max(0.2, min(3.0, new_agent.size_scale))

    # Inherit and mutate gait parameters
    new_agent.frequency = (parent1.frequency +
                           parent2.frequency) / 2 + random.uniform(-0.5, 0.5)
    new_agent.phase = (parent1.phase + parent2.phase) / 2 + random.uniform(
        -0.2, 0.2)

    if hasattr(parent1, "phase_offsets") and hasattr(parent2, "phase_offsets"):
      new_agent.phase_offsets = [
          (p1 + p2) / 2 + random.uniform(-0.2, 0.2)
          for p1, p2 in zip(parent1.phase_offsets, parent2.phase_offsets)
      ]

    # Update ID based on inherited parameters
    new_agent.update_id()

    # Rename it to include the hash!
    new_agent.name = f"{species_prefix}_{new_agent.id}"

    # Set position to fall from sky
    new_agent.pos = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 3.0]

    # Add to list
    self.agents.append(new_agent)

    # Set cooldowns
    parent1.cooldown = 5.0
    parent2.cooldown = 5.0
    new_agent.cooldown = 5.0

    # Reward parents for successful synthesis
    parent1.reward += 50.0
    parent2.reward += 50.0
    logger.info("Rewarded %s and %s for synthesis!", parent1.name, parent2.name)

    # Update positions of all existing agents for re-init
    for agent in self.agents[:-1]:
      body = self.data.body(agent.name)
      agent.pos = list(body.xpos)

    old_time = self.data.time
    logger.info("Re-initializing simulation with new agent: %s", new_agent.name)
    self.initialize()
    self.data.time = old_time

  def _save_agent_config(self, agent: Agent):
    """Saves the agent's configuration to a YAML file."""
    if isinstance(agent, ConfigurableAgent):
      agent_cfg = dict(agent.config)
      agent_cfg["frequency"] = float(agent.frequency)
      agent_cfg["phase"] = float(agent.phase)
      agent_cfg["amplitude"] = float(agent.amplitude)
      agent_cfg["leg_length_scale"] = float(agent.leg_length_scale)
      agent_cfg["phase_offsets"] = [float(p) for p in agent.phase_offsets]
      agent_cfg["parent_ids"] = getattr(agent, "parent_ids", [])
      agent_cfg["id"] = agent.id
      config = {"agents": [agent_cfg]}
    else:
      config = {
          "agents": [{
              "id": agent.id,
              "name": agent.name,
              "type": "default",
              "size_scale": float(agent.size_scale),
              "frequency": float(agent.frequency),
              "phase": float(agent.phase),
              "amplitude": float(agent.amplitude),
              "leg_length_scale": float(agent.leg_length_scale),
              "phase_offsets": [float(p) for p in agent.phase_offsets],
              "parent_ids": getattr(agent, "parent_ids", [])
          }]
      }

    os.makedirs("templates/agents", exist_ok=True)

    # Render an image of the agent
    try:
      renderer = mujoco.Renderer(self.model, 400, 400)
      renderer.update_scene(self.data, camera="main_cam")
      pixels = renderer.render()

      img_filename = f"templates/agents/{agent.name}.ppm"
      with open(img_filename, "wb") as f:
        f.write(f"P6\n400 400\n255\n".encode())
        f.write(pixels.tobytes())
      logger.info("Saved image of agent to %s", img_filename)
    except Exception as e:
      logger.error("Failed to render image for %s: %s", agent.name, e)

    filename = f"templates/agents/{agent.name}.yaml"
    with open(filename, "w") as f:
      yaml.dump(config, f)

    logger.info("Saved successful agent to %s", filename)

  def update_physics(self):
    """Updates physics parameters at runtime from the environment."""
    if self.model is None or self.data is None:
      raise RuntimeError(
          "Orchestrator not initialized. Call initialize() first.")

    self.env.update_runtime_physics(self.model, self.data)

  def get_state_dict(self) -> dict:
    """Returns a dictionary representing the current simulation state.

    Returns:
        dict: The state dictionary.
    """
    if self.data is None:
      return {"time": 0.0, "agents": {}}

    state = {"time": self.data.time, "agents": {}}

    for agent in self.agents:
      try:
        body = self.data.body(agent.name)
        state["agents"][agent.name] = {
            "pos": list(body.xpos),
            "color": agent.color,
            "reward": agent.reward,
            "cooldown": agent.cooldown,
        }
      except Exception as e:
        logger.error("Failed to get state for %s: %s", agent.name, e)

    return state

  def _respawn_agent(self, agent: Agent):
    """Resets the agent's position to fall from the sky."""
    if self.model is None or self.data is None:
      return

    # Find joint ID
    jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                               agent.name)
    if jnt_id < 0:
      logger.error("Failed to find joint for agent %s", agent.name)
      return

    # Find qpos address
    qpos_addr = self.model.jnt_qposadr[jnt_id]

    # Set new position
    new_pos = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 3.0]
    self.data.qpos[qpos_addr:qpos_addr + 3] = new_pos

    # Reset rotation (quaternion)
    self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

    # Reset velocities (qvel)
    dof_addr = self.model.jnt_dofadr[jnt_id]
    self.data.qvel[dof_addr:dof_addr + 6] = 0.0

    # Reset fallen time
    agent.fallen_time = 0.0

    logger.info("Respawned %s at %s", agent.name, new_pos)

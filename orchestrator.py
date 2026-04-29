"""Orchestrator module for MuJoCo simulation."""

import logging
import random
import xml.etree.ElementTree as ET

import mujoco

from agent import Agent
from environment import Environment

logger = logging.getLogger(__name__)


class Orchestrator:
  """Orchestrates the simulation, environment, and agents."""

  def __init__(self, env: Environment, agents: list[Agent], death_threshold: float = 3.0):
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
        actuator.append(act)

    return ET.tostring(root, encoding="unicode")

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

    # Check for death on fall
    for agent in self.agents:
      try:
        body = self.data.body(agent.name)
        if body.xpos[2] < 0.5:
          agent.fallen_time += self.model.opt.timestep
          if agent.fallen_time > self.death_threshold:
            logger.info("Agent %s died after falling for %.1f seconds", agent.name, self.death_threshold)
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

    # Create new agent
    i = len(self.agents)
    while True:
      new_name = f"agent_{i}"
      if not any(a.name == new_name for a in self.agents):
        break
      i += 1
    new_agent = Agent(name=new_name)
    new_agent.color = new_color

    # Set position to fall from sky
    new_agent.pos = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 3.0]

    # Add to list
    self.agents.append(new_agent)

    # Set cooldowns
    parent1.cooldown = 5.0
    parent2.cooldown = 5.0
    new_agent.cooldown = 5.0

    # Update positions of all existing agents for re-init
    for agent in self.agents[:-1]:
      body = self.data.body(agent.name)
      agent.pos = list(body.xpos)

    old_time = self.data.time
    logger.info("Re-initializing simulation with new agent: %s", new_name)
    self.initialize()
    self.data.time = old_time

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

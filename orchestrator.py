"""Orchestrator module for MuJoCo simulation."""

import logging
import math
import os
import random
import re
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
    self.total_syntheses = 0

    # Feature flags for demo safety.
    self.enable_synthesis = True
    self.enable_food = True
    self.enable_event_logging = False
    self.enable_flip_death = True
    self.respawn_occurred = False
    self.enable_respawn = True
    self.global_max_distance = 0.0

    # Game Mechanics: Food.
    self.food_positions = []
    for _ in range(30):
      self.food_positions.append(
          [random.uniform(-15.0, 15.0),
           random.uniform(-15.0, 15.0), 0.1])

  def log_event(self, event_type: str, agent_name: str, details: str = ""):
    """Logs a simulation event to a TSV file."""
    try:
      import os
      import time
      os.makedirs("logs", exist_ok=True)
      event_path = "logs/events.tsv"
      file_exists = os.path.exists(event_path)

      with open(event_path, "a") as f:
        if not file_exists:
          f.write("timestamp\tevent_type\tagent_name\tdetails\n")
        f.write(f"{time.time()}\t{event_type}\t{agent_name}\t{details}\n")
    except Exception as e:
      logger.error("Failed to log event: %s", e)

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

    # Add tracking camera for the first agent.
    if self.agents:
      ET.SubElement(
          worldbody,
          "camera",
          name="track_cam",
          mode="trackcom",
          target=self.agents[0].name,
          # Zoomed in closer to fill the frame.
          pos="0 -2 2",
          xyaxes="1 0 0 0 0.707 0.707")

    # Add food.
    for i, pos in enumerate(self.food_positions):
      food_body = ET.SubElement(
          worldbody,
          "body",
          name=f"food_{i}",
          pos=f"{pos[0]} {pos[1]} {pos[2]}",
      )
      ET.SubElement(food_body, "freejoint", name=f"food_{i}_joint")
      ET.SubElement(
          food_body,
          "geom",
          name=f"food_{i}_geom",
          type="sphere",
          size="0.1",
          rgba="1 0 0 1",  # Red food.
          contype="1",
          conaffinity="1",
      )

    # Add agents to worldbody
    for agent in self.agents:
      worldbody.append(agent.generate_xml())

    # Add actuators
    actuator = root.find("actuator")
    if actuator is None:
      actuator = ET.SubElement(root, "actuator")

    for agent in self.agents:
      if hasattr(agent, "dead") and agent.dead:
        continue
      for act in agent.generate_actuators_xml():
        act_name = act.get("name")
        logger.info("Adding actuator: %s", act_name)
        actuator.append(act)

    # Add sensors
    sensor = root.find("sensor")
    if sensor is None:
      sensor = ET.SubElement(root, "sensor")

    for agent in self.agents:
      for s in agent.generate_sensors_xml():
        sensor.append(s)

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

        # Decrease energy (hunger + effort)
        effort = 0.0
        if hasattr(agent, "limbs"):
          for limb in agent.limbs:
            m_name = f"{agent.name}_{limb['name']}_motor"
            m_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      m_name)
            if m_idx >= 0:
              effort += self.data.ctrl[m_idx]**2

            if "child" in limb:
              c_name = f"{agent.name}_{limb['child']['name']}_motor"
              c_idx = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_ACTUATOR, c_name)
              if c_idx >= 0:
                effort += self.data.ctrl[c_idx]**2
        else:
          # Fallback for base Agent with hardcoded legs
          for leg_name in ["left", "right"]:
            m_name = f"{agent.name}_{leg_name}_motor"
            m_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      m_name)
            if m_idx >= 0:
              effort += self.data.ctrl[m_idx]**2

        agent.energy -= (agent.hunger_rate +
                         0.1 * effort) * self.model.opt.timestep

        if agent.energy <= 0:
          logger.info("Agent %s died of starvation", agent.name)
          if self.enable_event_logging:
            self.log_event("death_starvation", agent.name)
          if self.enable_respawn:
            self._respawn_agent(agent)
          else:
            agent.dead = True
          continue

        # Find nearest food for this agent
        nearest_food_dist = float('inf')
        nearest_food_vec = [0.0, 0.0]

        try:
          body = self.data.body(agent.name)
          agent_pos = body.xpos

          if math.isnan(agent_pos[0]) or math.isnan(agent_pos[1]) or math.isnan(
              agent_pos[2]):
            logger.error("Agent %s has NaN position. Respawning.", agent.name)
            if self.enable_respawn:
              self._respawn_agent(agent)
            else:
              agent.dead = True
            continue

          for food_pos in self.food_positions:
            dist = ((agent_pos[0] - food_pos[0])**2 +
                    (agent_pos[1] - food_pos[1])**2)**0.5
            if dist < nearest_food_dist:
              nearest_food_dist = dist
              nearest_food_vec = [
                  food_pos[0] - agent_pos[0], food_pos[1] - agent_pos[1]
              ]

          # Store relative vector (normalized.)
          if nearest_food_dist > 0:
            agent.food_vector = [
                nearest_food_vec[0] / nearest_food_dist,
                nearest_food_vec[1] / nearest_food_dist
            ]
          else:
            agent.food_vector = [0.0, 0.0]
        except Exception:
          agent.food_vector = [0.0, 0.0]

        # Check for record distance
        try:
          dist = (agent_pos[0]**2 + agent_pos[1]**2)**0.5
          if dist > agent.max_distance:
            agent.max_distance = dist
          # Only log if it exceeds the global record by at least 1 meter.
          if dist > self.global_max_distance + 1.0:
            self.global_max_distance = dist
            logger.info("New record distance: %.2f meters by %s.", dist,
                        agent.name)
        except Exception:
          pass

        # Check food collisions
        if self.enable_food:
          agent_pos = body.xpos
          for i, food_pos in enumerate(self.food_positions):
            dist = ((agent_pos[0] - food_pos[0])**2 +
                    (agent_pos[1] - food_pos[1])**2)**0.5
            if dist < 0.5:
              agent.energy = min(agent.max_energy, agent.energy + 50.0)
              agent.food_eaten += 1
              # Increase frequency on eating food.
              agent.frequency = min(10.0, agent.frequency * 1.2)
              logger.info("Agent %s ate food %d!", agent.name, i)
              if self.enable_event_logging:
                self.log_event("eat_food", agent.name, f"food_{i}")
              # Respawn food.
              self.food_positions[i] = [
                  random.uniform(-10.0, 10.0),
                  random.uniform(-10.0, 10.0), 0.1
              ]
              # Re-initialize simulation to update food position.
              self._reinitialize_simulation()
              break  # Only eat one food per step.

        # Auto-save if survived 30 seconds and not saved yet.
        if agent.time_alive > 30.0 and not agent.saved:
          self._save_agent_config(agent)
          agent.saved = True

        # Drain health faster if on side or back (using quaternion to get z-component of z-axis).
        quat = body.xquat
        z_comp = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        if self.enable_flip_death and z_comp < 0.5:
          agent.health -= 50.0 * self.model.opt.timestep  # Drain fast.
          if agent.health <= 0:
            logger.info("Agent %s died of flipping (health depleted)",
                        agent.name)
            if self.enable_event_logging:
              self.log_event("death_flip", agent.name)
            # Mutate amplitude to get stronger.
            agent.amplitude = min(2.0, agent.amplitude * 1.2)
            if self.enable_respawn:
              self._respawn_agent(agent)
            else:
              agent.dead = True
            continue

        # Fall death threshold scaled by agent size.
        fall_threshold = 0.2 * agent.size_scale
        if body.xpos[2] < fall_threshold:
          agent.fallen_time += self.model.opt.timestep
          if agent.fallen_time > self.death_threshold:
            logger.info("Agent %s died after falling for %.1f seconds",
                        agent.name, self.death_threshold)
            if self.enable_event_logging:
              self.log_event("death_fall", agent.name)
            self._respawn_agent(agent)
        else:
          agent.fallen_time = 0.0
      except Exception as e:
        logger.error("Failed to check death for %s: %s", agent.name, e)

    # Check for synthesis
    self._check_synthesis()

    # Apply deferred mj_forward if any agent respawned!
    if self.respawn_occurred:
      mujoco.mj_forward(self.model, self.data)
      self.respawn_occurred = False

    # Periodic status update for the leader
    if self.data.time % 10.0 < self.model.opt.timestep:
      top_agent = None
      max_dist = -1.0
      for agent in self.agents:
        if hasattr(agent, "dead") and agent.dead:
          continue
        try:
          body = self.data.body(agent.name)
          dist = (body.xpos[0]**2 + body.xpos[1]**2)**0.5
          if dist > max_dist:
            max_dist = dist
            top_agent = agent
        except Exception:
          pass

      if top_agent:
        logger.info("Leader: %s, Distance: %.2f, Energy: %.1f.", top_agent.name,
                    max_dist, top_agent.energy)

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

      # Ignore food collisions for synthesis.
      if geom1_name.startswith("food_") or geom2_name.startswith("food_"):
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
        # Collision between different agents.
        if agent1.cooldown <= 0 and agent2.cooldown <= 0:
          logger.info("Synthesis triggered between %s and %s", agent1.name,
                      agent2.name)
          if self.enable_event_logging:
            self.log_event("synthesis", agent1.name, f"with_{agent2.name}")
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

    if isinstance(parent1, ConfigurableAgent) and isinstance(parent2, ConfigurableAgent):
      # Randomly pick one parent's morphology
      parent = random.choice([parent1, parent2])
      new_agent = ConfigurableAgent(name="temp_agent", config=parent.config)
    elif isinstance(parent1, ConfigurableAgent):
      new_agent = ConfigurableAgent(name="temp_agent", config=parent1.config)
    elif isinstance(parent2, ConfigurableAgent):
      new_agent = ConfigurableAgent(name="temp_agent", config=parent2.config)
    else:
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

    # Rename it to include the hash.
    new_agent.name = f"{species_prefix}_{new_agent.id}"

    # Set position to fall from sky
    new_agent.pos = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 3.0]

    # Add to list
    self.agents.append(new_agent)
    self.total_syntheses += 1

    # Set cooldowns
    parent1.cooldown = 5.0
    parent2.cooldown = 5.0
    new_agent.cooldown = 5.0

    # Reward parents for successful synthesis
    parent1.reward += 50.0
    parent1.syntheses_count += 1
    parent2.reward += 50.0
    parent2.syntheses_count += 1
    logger.info("Rewarded %s and %s for synthesis!", parent1.name, parent2.name)

    logger.info("Re-initializing simulation for new agent: %s", new_agent.name)
    self._reinitialize_simulation()

  def _reinitialize_simulation(self):
    """Re-initializes the simulation while preserving agent states."""
    for agent in self.agents:
      try:
        body = self.data.body(agent.name)
        agent.pos = list(body.xpos)
      except Exception:
        pass  # Ignore agents not in simulation.

    old_time = self.data.time
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

    species = agent.name.split("__")[0] if "__" in agent.name else re.sub(
        r"_[0-9a-f]+$", "", agent.name)
    species = re.sub(r"_default$", "", species)

    species_dir = f"templates/agents/{species}"
    os.makedirs(species_dir, exist_ok=True)

    # Image rendering removed as per user request.

    filename = f"{species_dir}/{agent.name}.yaml"
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

    # Set new position closer to ground to prevent flipping on landing
    new_pos = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 0.5]
    self.data.qpos[qpos_addr:qpos_addr + 3] = new_pos

    # Reset rotation (quaternion)
    self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

    # Reset velocities.
    dof_addr = self.model.jnt_dofadr[jnt_id]
    self.data.qvel[dof_addr:dof_addr + 6] = 0.0

    # Flag that respawn occurred to call mj_forward once at the end of step
    self.respawn_occurred = True

    # Mutate agent slightly on respawn to get unstuck.
    agent.frequency = max(0.5,
                          min(10.0, agent.frequency * random.uniform(0.9, 1.1)))
    agent.phase += random.uniform(-0.2, 0.2)

    # Reset fallen time and energy
    agent.fallen_time = 0.0
    agent.energy = agent.max_energy

    logger.info("Respawned %s at %s", agent.name, new_pos)

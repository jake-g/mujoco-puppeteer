"""Unit tests for the orchestrator module."""

import unittest
import xml.etree.ElementTree as ET

import mujoco

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator


class TestOrchestrator(unittest.TestCase):

  def setUp(self):
    self.env = Environment()
    self.agent1 = Agent(name="agent_1")
    self.agent2 = Agent(name="agent_2")
    self.orchestrator = Orchestrator(self.env, [self.agent1, self.agent2])

  def test_combine_xml(self):
    """Test that XML is combined correctly."""
    xml_str = self.orchestrator.generate_combined_xml()
    root = ET.fromstring(xml_str)

    # Check environment elements
    self.assertIsNotNone(root.find("option"))
    self.assertIsNotNone(root.find("asset"))

    # Check agent elements in worldbody
    worldbody = root.find("worldbody")
    agent_names = [
        child.get("name") for child in worldbody if child.tag == "body"
    ]
    self.assertIn("agent_1", agent_names)
    self.assertIn("agent_2", agent_names)

    # Check actuators
    actuator = root.find("actuator")
    self.assertIsNotNone(actuator)
    actuator_names = [child.get("name") for child in actuator]
    self.assertIn("agent_1_left_motor", actuator_names)
    self.assertIn("agent_2_left_motor", actuator_names)

  def test_orchestrator_step(self):
    """Test that step() advances time and updates rewards."""
    self.orchestrator.initialize()

    initial_time = self.orchestrator.data.time
    self.orchestrator.step()
    final_time = self.orchestrator.data.time

    self.assertGreater(final_time, initial_time)

    # Check that rewards are updated
    self.assertIn("agent_1", self.orchestrator.rewards)
    self.assertIn("agent_2", self.orchestrator.rewards)

  def test_runtime_parameter_change(self):
    """Test changing gravity at runtime."""
    self.orchestrator.initialize()

    # Initial gravity
    self.assertEqual(self.orchestrator.model.opt.gravity[2], -9.81)

    # Change via environment
    self.env.set_gravity([0.0, 0.0, -5.0])
    self.orchestrator.update_physics()

    # Check if model updated
    self.assertEqual(self.orchestrator.model.opt.gravity[2], -5.0)

  def test_zero_agents(self):
    """Test orchestrator with zero agents (empty world)."""
    orchestrator = Orchestrator(self.env, [])
    orchestrator.initialize()
    self.assertEqual(len(orchestrator.rewards), 0)

  def test_many_agents(self):
    """Test orchestrator with many agents (stress test)."""
    agents = [Agent(name=f"agent_{i}") for i in range(10)]
    orchestrator = Orchestrator(self.env, agents)
    orchestrator.initialize()
    self.assertEqual(len(orchestrator.rewards), 10)

    # Verify all agents are in the XML
    xml_str = orchestrator.generate_combined_xml()
    for i in range(10):
      self.assertIn(f"agent_{i}", xml_str)

  def test_get_state_dict(self):
    """Test that state dictionary is generated correctly."""
    self.orchestrator.initialize()
    state = self.orchestrator.get_state_dict()

    self.assertIn("time", state)
    self.assertIn("agents", state)

    # Check agent_1 state
    self.assertIn("agent_1", state["agents"])
    self.assertIn("pos", state["agents"]["agent_1"])
    self.assertIn("color", state["agents"]["agent_1"])


  def test_synthesis(self):
    """Test that synthesis creates a valid hybrid agent."""
    # Create configurable agents
    cfg = {"type": "test_species", "limbs": [{"name": "torso", "size": [0.2]}]}
    parent1 = ConfigurableAgent(name="parent_1", config=cfg)
    parent2 = ConfigurableAgent(name="parent_2", config=cfg)

    orchestrator = Orchestrator(self.env, [parent1, parent2])
    orchestrator.initialize()
    orchestrator.enable_export = False

    # Trigger synthesis
    orchestrator._synthesize_agents(parent1, parent2)

    # Check that a new agent was added
    self.assertEqual(len(orchestrator.agents), 3)
    new_agent = orchestrator.agents[2]

    # Check that it is a ConfigurableAgent
    self.assertIsInstance(new_agent, ConfigurableAgent)
    # Check that it inherited the config!
    self.assertEqual(new_agent.config["type"], "test_species")


if __name__ == "__main__":
  unittest.main()

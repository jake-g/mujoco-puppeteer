"""Unit tests for the agent module."""

import unittest
import xml.etree.ElementTree as ET

import mujoco

from agent import Agent
from agent import ConfigurableAgent
from environment import Environment


class TestAgent(unittest.TestCase):

  def setUp(self):
    self.agent = Agent(name="test_agent")

  def test_generate_xml(self):
    """Test that XML is generated and contains key elements."""
    element = self.agent.generate_xml()
    self.assertEqual(element.tag, "body")
    self.get_name = element.get("name")
    self.assertEqual(self.get_name, "test_agent")

    # Check for children
    tags = [child.tag for child in element]
    self.assertIn("freejoint", tags)
    self.assertIn("geom", tags)

  def test_generate_actuators_xml(self):
    """Test that actuators XML is generated."""
    actuators = self.agent.generate_actuators_xml()
    self.assertEqual(len(actuators), 2)
    self.assertEqual(actuators[0].tag, "motor")
    self.assertEqual(actuators[0].get("joint"), "test_agent_left_hip")

  def test_calculate_reward(self):
    """Test reward calculation."""
    # To test this properly, we need a compiled model containing the agent.
    # Let's use the Environment class to help us build it.
    env = Environment()

    # We need to combine them. Let's do it manually here for the test.
    env_xml = env.generate_xml()
    root = ET.fromstring(env_xml)

    worldbody = root.find("worldbody")
    worldbody.append(self.agent.generate_xml())

    actuator = ET.SubElement(root, "actuator")
    for act in self.agent.generate_actuators_xml():
      actuator.append(act)

    combined_xml = ET.tostring(root, encoding="unicode")

    model = mujoco.MjModel.from_xml_string(combined_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    test_cases = [
        ("initial", [0.0, 0.0, 1.0], 0.1),
        ("fall", [0.0, 0.0, 0.4], -10.0),
        ("forward", [5.0, 0.0, 1.0], 5.1),
    ]

    for name, pos, expected_reward in test_cases:
      with self.subTest(name=name):
        data.body("test_agent").xpos[0] = pos[0]
        data.body("test_agent").xpos[1] = pos[1]
        data.body("test_agent").xpos[2] = pos[2]
        reward = self.agent.calculate_reward(data)
        self.assertEqual(reward, expected_reward)


class TestConfigurableAgent(unittest.TestCase):

  def setUp(self):
    self.config = {
        "body": {
            "type": "box",
            "size": [0.2, 0.2, 0.05]
        },
        "limbs": [{
            "name": "leg1",
            "pos": [0.1, 0.1, 0],
            "axis": [0, 1, 0],
            "range": [-30, 30],
            "geom": {
                "type": "capsule",
                "size": [0.02, 0.1]
            }
        }, {
            "name": "leg2",
            "pos": [-0.1, -0.1, 0],
            "axis": [0, 1, 0],
            "range": [-30, 30],
            "geom": {
                "type": "capsule",
                "size": [0.02, 0.1]
            },
            "child": {
                "name": "calf2",
                "axis": [0, 1, 0],
                "range": [0, 90],
                "geom": {
                    "type": "capsule",
                    "size": [0.015, 0.1]
                }
            }
        }]
    }
    self.agent = ConfigurableAgent(name="test_config", config=self.config)

  def test_generate_xml(self):
    """Test XML generation with child limbs."""
    element = self.agent.generate_xml()
    self.assertEqual(element.tag, "body")

    # Find limb bodies
    limbs = element.findall("body")
    self.assertEqual(len(limbs), 2)

    # Check child limb in leg2
    leg2 = element.find(".//body[@name='test_config_leg2_leg']")
    self.assertIsNotNone(leg2)
    child = leg2.find("body")
    self.assertIsNotNone(child)
    self.assertEqual(child.get("name"), "test_config_calf2_calf")

  def test_generate_actuators(self):
    """Test that it creates actuators for both hip and knee."""
    actuators = self.agent.generate_actuators_xml()
    # 2 limbs, one has child, so 2 + 1 = 3 actuators!
    self.assertEqual(len(actuators), 3)

    names = [act.get("name") for act in actuators]
    self.assertIn("test_config_leg1_motor", names)
    self.assertIn("test_config_leg2_motor", names)
    self.assertIn("test_config_calf2_motor", names)


if __name__ == "__main__":
  unittest.main()

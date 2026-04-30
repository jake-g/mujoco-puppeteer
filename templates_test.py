"""Unit tests for the templates."""

import os
import unittest

import mujoco
import yaml

from agent import ConfigurableAgent
from environment import Environment
from orchestrator import Orchestrator


class TestTemplates(unittest.TestCase):

  def setUp(self):
    self.templates_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "templates"))

  def test_base_template_valid(self):
    """Test that base.yaml is valid YAML and has correct structure."""
    base_path = os.path.join(self.templates_dir, "scenes", "base.yaml")
    self.assertTrue(os.path.exists(base_path))

    with open(base_path, "r") as f:
      config = yaml.safe_load(f)

    self.assertIn("environment", config)
    self.assertIn("agents", config)
    self.assertIsInstance(config["agents"], list)

  def test_all_templates_structure(self):
    """Test that all templates follow the base structure."""
    # Test agent templates
    agents_dir = os.path.join(self.templates_dir, "agents")
    if os.path.exists(agents_dir):
      for root, _, files in os.walk(agents_dir):
        if "old" in root:
          continue
        for filename in files:
          if filename.endswith(".yaml"):
            path = os.path.join(root, filename)
            with self.subTest(path=path):
              with open(path, "r") as f:
                config = yaml.safe_load(f)
              self.assertTrue("environment" in config or "agents" in config)

    # Test scene templates
    scenes_dir = os.path.join(self.templates_dir, "scenes")
    if os.path.exists(scenes_dir):
      for filename in os.listdir(scenes_dir):
        if filename.endswith(".yaml"):
          path = os.path.join(scenes_dir, filename)
          with self.subTest(path=path):
            with open(path, "r") as f:
              config = yaml.safe_load(f)
            self.assertTrue("environment" in config or "agents" in config)

  def test_templates_compilation(self):
    """Test that all agent templates compile successfully."""
    agents_dir = os.path.join(self.templates_dir, "agents")
    if not os.path.exists(agents_dir):
      return

    for root, _, files in os.walk(agents_dir):
      if "old" in root:
        continue
      for filename in files:
        if filename.endswith(".yaml"):
          path = os.path.join(root, filename)
          with self.subTest(path=path):
            try:
              with open(path, "r") as file:
                config = yaml.safe_load(file)

              env = Environment()
              agents = []
              if "agents" in config:
                for agent_cfg in config["agents"]:
                  agent = ConfigurableAgent(name=agent_cfg["name"],
                                            config=agent_cfg)
                  agents.append(agent)

              orch = Orchestrator(env, agents)
              xml_str = orch.generate_combined_xml()

              # Try to compile model
              mujoco.MjModel.from_xml_string(xml_str)
            except Exception as e:
              self.fail(f"{path} failed to compile: {e}")

  def test_limb_connections(self):
    """Check if limbs are close enough to their parents to be connected."""
    agents_dir = os.path.join(self.templates_dir, "agents")
    if not os.path.exists(agents_dir):
      return

    for root, _, files in os.walk(agents_dir):
      if "old" in root:
        continue
      for filename in files:
        if filename.endswith(".yaml"):
          path = os.path.join(root, filename)
          with self.subTest(path=path):
            with open(path, "r") as file:
              config = yaml.safe_load(file)

            if "agents" in config:
              for agent_cfg in config["agents"]:
                if "limbs" in agent_cfg:
                  limbs = agent_cfg["limbs"]
                  # Find torso or root body
                  torso = None
                  for limb in limbs:
                    if limb.get("name") in ["torso", "body"]:
                      torso = limb
                      break

                  if torso and "size" in torso:
                    t_size = torso["size"]
                    for limb in limbs:
                      if limb == torso:
                        continue
                      if "pos" in limb:
                        pos = limb["pos"]
                        # Check if distance is not too large
                        dist = sum(p**2 for p in pos)**0.5
                        max_t_dim = max(t_size)
                        self.assertLess(
                            dist, max_t_dim * 3.0,
                            f"Limb {limb['name']} seems too far from torso in {path}"
                        )


if __name__ == "__main__":
  unittest.main()

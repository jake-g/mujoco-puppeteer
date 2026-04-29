"""Unit tests for the templates."""

import os
import unittest
import yaml


class TestTemplates(unittest.TestCase):

  def setUp(self):
    self.templates_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "templates")
    )

  def test_base_template_valid(self):
    """Test that base.yaml is valid YAML and has correct structure."""
    base_path = os.path.join(self.templates_dir, "base.yaml")
    self.assertTrue(os.path.exists(base_path))

    with open(base_path, "r") as f:
      config = yaml.safe_load(f)

    self.assertIn("environment", config)
    self.assertIn("agents", config)
    self.assertIsInstance(config["agents"], list)

  def test_all_templates_structure(self):
    """Test that all templates in the folder follow the base structure."""
    for filename in os.listdir(self.templates_dir):
      if filename.endswith(".yaml"):
        with self.subTest(filename=filename):
          path = os.path.join(self.templates_dir, filename)
          with open(path, "r") as f:
            config = yaml.safe_load(f)

          # Every template should at least have environment or agents
          self.assertTrue("environment" in config or "agents" in config)

          if "environment" in config:
            env = config["environment"]
            # If gravity is defined, it should be a list of 3 floats
            if "gravity" in env:
              self.assertEqual(len(env["gravity"]), 3)

          if "agents" in config:
            agents = config["agents"]
            self.assertIsInstance(agents, list)
            for agent in agents:
              self.assertIn("name", agent)


if __name__ == "__main__":
  unittest.main()

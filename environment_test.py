"""Unit tests for the environment module."""

import unittest

import mujoco

from environment import Environment


class TestEnvironment(unittest.TestCase):

  def setUp(self):
    self.env = Environment()

  def test_generate_xml(self):
    """Test that XML is generated and contains key elements."""
    xml_str = self.env.generate_xml()
    self.assertIn("<mujoco", xml_str)
    self.assertIn('builtin="checker"', xml_str)
    self.assertIn('name="floor"', xml_str)

  def test_create_model(self):
    """Test that the model can be created from XML."""
    model = self.env.create_model()
    self.assertIsNotNone(model)
    self.assertIsInstance(model, mujoco.MjModel)

  def test_set_gravity(self):
    """Test setting gravity."""
    new_gravity = [0.0, 0.0, -5.0]
    self.env.set_gravity(new_gravity)
    self.assertEqual(self.env.gravity, new_gravity)

    xml_str = self.env.generate_xml()
    self.assertIn('gravity="0.0 0.0 -5.0"', xml_str)

  def test_update_runtime_physics(self):
    """Test updating physics at runtime."""
    model = self.env.create_model()
    data = mujoco.MjData(model)

    # Initial gravity should be default
    self.assertEqual(model.opt.gravity[2], -9.81)

    # Change gravity in python object
    new_gravity = [0.0, 0.0, -5.0]
    self.env.set_gravity(new_gravity)

    # Update runtime
    self.env.update_runtime_physics(model, data)

    # Check if model updated
    self.assertEqual(model.opt.gravity[2], -5.0)

  def test_custom_floor_colors(self):
    """Test generating XML with custom floor colors (e.g., neon grid)."""
    self.env.floor_rgb1 = [0.0, 1.0, 0.0]  # Neon Green
    self.env.floor_rgb2 = [0.0, 0.0, 1.0]  # Neon Blue
    xml_str = self.env.generate_xml()
    self.assertIn('rgb1="0.0 1.0 0.0"', xml_str)
    self.assertIn('rgb2="0.0 0.0 1.0"', xml_str)

  def test_zero_gravity(self):
    """Test environment in zero gravity (space mode)."""
    self.env.set_gravity([0.0, 0.0, 0.0])
    model = self.env.create_model()
    self.assertEqual(model.opt.gravity[2], 0.0)

  def test_skybox_colors(self):
    """Test generating XML with custom skybox colors."""
    self.env.sky_rgb1 = [1.0, 0.0, 0.0]  # Red sky
    self.env.sky_rgb2 = [0.0, 1.0, 0.0]  # Green horizon
    xml_str = self.env.generate_xml()
    self.assertIn('rgb1="1.0 0.0 0.0"', xml_str)
    self.assertIn('rgb2="0.0 1.0 0.0"', xml_str)

  def test_wind(self):
    """Test setting wind."""
    self.env.wind = [1.0, 0.0, 0.0]
    xml_str = self.env.generate_xml()
    self.assertIn('wind="1.0 0.0 0.0"', xml_str)

  def test_update_runtime_wind(self):
    """Test updating wind at runtime."""
    model = self.env.create_model()
    data = mujoco.MjData(model)

    # Initial wind should be zero
    self.assertEqual(model.opt.wind[0], 0.0)

    # Change wind in python object
    self.env.wind = [1.0, 0.0, 0.0]

    # Update runtime
    self.env.update_runtime_physics(model, data)

    # Check if model updated
    self.assertEqual(model.opt.wind[0], 1.0)


if __name__ == "__main__":
  unittest.main()

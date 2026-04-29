"""Unit tests for the client module."""

import unittest
import json
from client import process_message


class TestClient(unittest.TestCase):

  def test_process_message_valid(self):
    """Test processing a valid JSON message."""
    valid_message = json.dumps({"time": 1.0, "agents": {}})
    state = process_message(valid_message)
    self.assertEqual(state["time"], 1.0)
    self.assertEqual(state["agents"], {})

  def test_process_message_invalid(self):
    """Test processing an invalid JSON message."""
    invalid_message = "invalid json"
    state = process_message(invalid_message)
    self.assertEqual(state, {})


if __name__ == "__main__":
  unittest.main()

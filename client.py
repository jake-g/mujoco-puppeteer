"""WebSocket client for receiving MuJoCo simulation state."""

import asyncio
import json
import logging
import os
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_message(message: str) -> dict:
  """Processes a received WebSocket message.

  Args:
      message: The raw string message.

  Returns:
      dict: The decoded state dictionary, or empty dict on failure.
  """
  try:
    state = json.loads(message)
    return state
  except json.JSONDecodeError:
    logger.warning("Failed to decode JSON message")
    return {}


async def listen(uri: str):
  """Connects to the server and listens for state updates.

  Args:
      uri: The WebSocket URI to connect to.
  """
  async with websockets.connect(uri) as websocket:
    logger.info("Connected to %s", uri)

    try:
      async for message in websocket:
        state = process_message(message)
        if state:
          logger.info("Received state: Time=%.2f, Agents=%d", state["time"],
                      len(state["agents"]))
          if state["agents"]:
            first_agent = list(state["agents"].keys())[0]
            logger.info("  %s pos: %s", first_agent,
                        state["agents"][first_agent]["pos"])
    except websockets.ConnectionClosed:
      logger.info("Connection closed.")


if __name__ == "__main__":
  uri = os.environ.get("SIM_URI", "ws://localhost:8765")
  try:
    asyncio.run(listen(uri))
  except KeyboardInterrupt:
    logger.info("Client stopped.")

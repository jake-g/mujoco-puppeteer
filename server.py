"""WebSocket server for streaming MuJoCo simulation state."""

import asyncio
import json
import logging
import os
import time
import websockets
from environment import Environment
from agent import Agent
from orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimulationServer:
  """Manages the WebSocket server and simulation loop."""

  def __init__(self, host: str = None, port: int = None):
    self.host = host or os.environ.get("SIM_HOST", "localhost")
    self.port = port or int(os.environ.get("SIM_PORT", 8765))
    self.env = Environment()
    self.agent1 = Agent(name="agent_1")
    self.agent2 = Agent(name="agent_2")
    self.orchestrator = Orchestrator(self.env, [self.agent1, self.agent2])
    self.clients = set()

  async def handler(self, websocket):
    """Handles incoming WebSocket connections."""
    logger.info("New client connected: %s", websocket.remote_address)
    self.clients.add(websocket)
    try:
      async for message in websocket:
        logger.info("Received message from %s: %s", websocket.remote_address,
                    message)
        try:
          data = json.loads(message)
          if "action" in data:
            logger.info("Action requested: %s", data["action"])
        except json.JSONDecodeError:
          logger.warning("Failed to decode JSON message")
    except websockets.ConnectionClosed:
      logger.info("Client disconnected: %s", websocket.remote_address)
    finally:
      self.clients.remove(websocket)

  async def broadcast_state(self):
    """Broadcasts the simulation state to all connected clients."""
    while True:
      if self.orchestrator.data:
        state = self.orchestrator.get_state_dict()
        message = json.dumps(state)

        if self.clients:
          await asyncio.gather(
              *[client.send(message) for client in self.clients],
              return_exceptions=True)

      await asyncio.sleep(0.05)  # 20Hz update rate

  async def run_simulation_loop(self):
    """Runs the physics simulation loop."""
    self.orchestrator.initialize()
    logger.info("Simulation initialized.")

    while True:
      step_start = time.time()

      # Step simulation
      self.orchestrator.step()

      # Rudimentary time keeping to match physics time
      # Default timestep is usually 0.002
      time_until_next_step = self.orchestrator.model.opt.timestep - (
          time.time() - step_start)
      if time_until_next_step > 0:
        await asyncio.sleep(time_until_next_step)
      else:
        # If we are falling behind, yield to allow other tasks to run
        await asyncio.sleep(0)

  async def start(self):
    """Starts the server and simulation loops."""
    async with websockets.serve(self.handler, self.host, self.port):
      logger.info("Server started at ws://%s:%d", self.host, self.port)

      # Run both loops concurrently
      await asyncio.gather(self.broadcast_state(), self.run_simulation_loop())


if __name__ == "__main__":
  server = SimulationServer()
  try:
    asyncio.run(server.start())
  except KeyboardInterrupt:
    logger.info("Server stopped.")

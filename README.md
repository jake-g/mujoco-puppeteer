# mujoco-puppeteer

A live-streamed, multiplayer game or simulation where viewers can observe a MuJoCo simulation and take turns controlling or modifying it.

## Overview

The goal of this project is to create an interactive environment where a physics simulation runs (powered by MuJoCo) and external players can connect via a minimal network layer to observe and interact with it. Think of it as a crowd-controlled simulation or a "Truman Show" for AI agents, where the audience can play "God" and manipulate forces, gravity, or terrain.

## System Architecture

This project implements a server-client architecture for a live-streamed, multiplayer physics simulation.

### Local Simulation & Orchestration
-   **Orchestrator**: The `orchestrator.py` module combines the generated XML from `environment.py` and `agent.py` into a single MuJoCo model.
-   **Visualization**: On macOS, the simulation is executed via the `mjpython` launcher to support the passive viewer (`launch_passive`) without thread conflicts.

### Multiplayer & Streaming (The "Puppeteer" System)
*   **Simulation Server**: The `server.py` script runs the authoritative MuJoCo simulation and broadcasts the state vector to connected clients.
*   **State Streaming**: Instead of video, the server streams raw state vectors (positions, velocities, forces) via WebSockets. This minimizes bandwidth and allows clients to render locally.
*   **Clients**: The `client.py` script demonstrates connecting to the server and receiving state updates.
## Status & Milestones

-   Foundation, Environment, Agent, and Orchestration are fully implemented and tested.
-   Visualization is implemented using `launch_passive` and `mjpython` for Mac compatibility. Interactive keyboard controls are available in `cli.py`.
-   Networking is started with state streaming via WebSockets in `server.py` and `client.py`.
-   Learning & Evolution is started with a Genetic Algorithm in `train.py`.

## Simulation Mechanics

-   **Synthesis**: When two agents collide, they synthesize and create a new agent.
    -   **Color Mixing & Mutation**: The new agent's color is the average of the parents' colors, with a slight random mutation to create variety.
    -   **Spawn Effect**: The new agent spawns at a height and falls to the ground at a random position on the floor.
    -   **Cooldown**: Agents have a cooldown period after synthesis to prevent overpopulation.
    -   **Dynamic Re-initialization**: The simulation dynamically re-initializes to add the new agent while preserving the positions and time of existing agents.
-   **Agent Scaling**: Agents support a `size_scale` parameter in the configuration, allowing for "Giant" and "Dwarf" variants that scale the body and legs proportionally.
-   **Death on Fall**: If an agent remains below a height of 0.5 for more than 3 seconds, it "dies" and is respawned falling from the sky at a random position.
-   **Quadruped Variant**: A 4-legged agent variant (`QuadrupedAgent`) is available for better stability and complex gaits.
-   **Learning & Evolution**: Run `train.py` to evolve agent walking parameters (frequency, phase) using a Genetic Algorithm. The best results are saved to `templates/evolved_quadruped.yaml`.

## Interactive Controls

When running the simulation via `cli.py` (or `make run-template`), you can use the following keyboard controls in the viewer window:

-   **Spacebar**: Pause/Resume simulation.
-   **G**: Invert gravity (flips direction).

## Codebase Structure

The project uses a flat directory structure for Python files to keep it simple.

- [environment.py](./environment.py): Defines the `Environment` class, managing the world and physics settings.
- [agent.py](./agent.py): Defines the `Agent` class, representing simulated entities with reward functions.
- [environment_test.py](./environment_test.py): Unit tests for the environment module.
- [agent_test.py](./agent_test.py): Unit tests for the agent module.
- [orchestrator.py](./orchestrator.py): Defines the `Orchestrator` class, combining environment and agents.
- [orchestrator_test.py](./orchestrator_test.py): Unit tests for the orchestrator module.
- [simulate_visual.py](./simulate_visual.py): Runs the simulation with a visual window and runtime interactions.
- [server.py](./server.py): WebSocket server for streaming simulation state.
- [client.py](./client.py): WebSocket client for receiving state updates.
- [cli.py](./cli.py): CLI for selecting and launching simulation templates.
- [templates/](./templates/): Folder containing YAML simulation templates.
- [train.py](./train.py): Genetic Algorithm for evolving agent policies.
- [Makefile](./Makefile): Manages setup, formatting, and tests.
- [DEV_LOG.md](./DEV_LOG.md): Log of notable contributions and milestones.

## Development Commands

This project uses a `Makefile` to manage development tasks.

-   `make setup`: Set up the virtual environment and install dependencies.
-   `make format`: Run YAPF and pre-commit checks.
-   `make test`: Run all unit tests.
-   `make run`: Run the local visual simulation (requires Mac GUI environment).
-   `make server`: Run the WebSocket simulation server in the foreground.
-   `make server-bg`: Run the server in the background (logs to `logs/server.log`).
-   `make server-stop`: Stop the background server.
-   `make client`: Run the test client to observe state streaming.
-   `make list`: List available simulation templates.
-   `make run-template name=<template_name>`: Run a specific template (e.g., `make run-template name=neon_grid`).
-   `make clean`: Clean up logs and cache files.

## Coding Style

This project follows the **Google Python Style Guide**, consistent with the MuJoCo source library (as seen in `mujoco_src/python/setup.py`).
*   **Indentation**: 2 spaces.
*   **Line Length**: Maximum 80 characters.
*   **Tools**: `yapf` and `isort` are used for formatting, enforced via `pre-commit`.

## MuJoCo Submodule

This project includes the MuJoCo physics engine as a submodule.

*   **Repository**: [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
*   **Documentation**: [MuJoCo Documentation](https://mujoco.readthedocs.io/)

### Important Details

MuJoCo stands for Multi-Joint dynamics with Contact. It is a data-oriented physics engine designed for robotics, biomechanics, and machine learning. Key features include:
*   Simulation in generalized coordinates combined with optimization-based contact dynamics.
*   Separation of model description (`mjModel`) and simulation data (`mjData`).
*   Zero memory allocations during runtime after initialization.

## Future Work

With the core foundation, networking, and basic evolution in place, future development will focus on advanced gait learning and scaling up to multiplayer interaction.

### Advanced Gait Evolution (Inspired by [evolution-sim](https://github.com/jake-g/evolution-sim))
-   **Amplitude Evolution**: Expand the genome to include amplitude (`cosFactor`) for each joint, as seen in the 2D simulation.
-   **Genetic Diversity**: Implement single-point crossover and multiplicative mutation to better explore the parameter space.
-   **Step-Based Rewards**: Implement step detection (tracking alternate ground contact) to reward actual walking behavior rather than just translation.

### Full Humanoid Simulation
-   **Complex Structure**: Move from Quadruped to a full Humanoid structure (head, neck, torso, arms, legs) inspired by the 2D simulation's `human.js`.
-   **Actuator Complexity**: Handle multi-DOF joints to allow for realistic human-like movement.

### Deep Reinforcement Learning
-   **Policy Optimization**: Move beyond simple sine waves and integrate deep RL libraries (e.g., Stable Baselines3) to learn complex, robust policies for walking and interaction.

### Advanced Multiplayer Interaction
-   **Bidirectional Control**: Implement control inputs from clients to the server (e.g., applying external forces or "pushing" agents).
-   **Permissions & Roles**: Implement account-based permissions to gate access to controls (Observer vs. Puppeteer vs. Director).

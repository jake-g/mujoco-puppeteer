# mujoco-puppeteer

A live-streamed, multiplayer game or simulation where viewers can observe a MuJoCo simulation and take turns controlling or modifying it.

## Quick Start

Get up and running with the simulation in seconds:

1.  **Setup Environment**: Install dependencies and setup the virtual environment.
    ```bash
    make setup
    ```
2.  **Run Demo**: Launch the visual demo to see evolved agents interacting.
    ```bash
    make demo
    ```
    Or run and record frames to generate GIFs (auto-compiles to GIF on Ctrl+C):
    ```bash
    make demo-record
    ```
3.  **Run Specific Scene**: Load a specific scene template.
    ```bash
    make run-template name=chaos_colosseum
    ```

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

### Goal Seeking & Steering
- **Virtual Chemotaxis**: Agents compute a normalized relative position vector to the closest food target, allowing closed-loop goal seeking.
- **Asymmetric Torque Modulation**: Turning is achieved by increasing amplitude on the side contralateral to the target.

### Fatigue & Excitement
- **Fatigue Simulation**: Actuator amplitude scales linearly with agent energy to simulate exhaustion.
- **Food Excitement**: Control frequency increases by 20% when an agent successfully eats food.

### Results & Telemetry
- **Simplified Index**: `results/index.yaml` is reduced to a clean folder-to-file-count mapping.
- **Species Censuses**: Maintenance tracks total variant counts per species (e.g., total Gorillas vs. total Snakes).
- **Smooth Playback**: Evolution GIFs are compiled at $20\text{ fps}$ ($50\text{ms}$ intervals) to make analyzing movement gaits easier on the eyes.

## Simulation Mechanics

-   **Synthesis**: When two agents collide, they synthesize and create a new agent.
    -   **Conditions**: Both agents must have their cooldown at 0.
    -   **Inheritance**: The new agent's color is the average of the parents' colors. Gait parameters (frequency, phase, offsets) are averaged and mutated.
    -   **Cross-Synthesis**: Different species CAN synthesize! They create a hybrid named `[Parent1]_[Parent2]_Hybrid`. Currently, they inherit the physical structure of Parent 1.
    -   **Spawn Effect**: The new agent spawns at a height and falls to the ground at a random position on the floor.
    -   **Cooldown**: Agents have a cooldown period after synthesis to prevent overpopulation.
-   **Hunger & Starvation**: Agents have energy that decreases over time.
    -   **Hunger Rate**: Scaled by agent size (bigger agents starve faster!).
    -   **Starvation**: If energy reaches 0, the agent dies and respawns.
-   **Food**: Red spheres spawn randomly on the map.
    -   **Eating**: If an agent gets close to food, it consumes it and recovers 50 energy.
-   **Death on Fall**: If an agent remains below a height threshold (scaled by size) for more than 3 seconds, it "dies" and is respawned falling from the sky.
-   **Configurable Agent**: An abstract agent (`ConfigurableAgent`) that can represent any creature (Turtle, Hexapod, Biped, etc.) by defining its body and limbs in YAML!
-   **Step Detection Reward**: Quadruped agents reward alternating ground contact by feet, giving a bonus for each step to encourage true walking.
-   **Learning & Evolution**: Run `train.py` to evolve agent walking parameters using a Genetic Algorithm.
-   **Automated Evolution**: Run `auto_evolve.py` to evolve multiple species sequentially and save their best configurations.
-   **Leaderboard**: Run `update_leaderboard.py` to evaluate all templates and rank them in `LEADERBOARD.md`.
-   **Per-Species Evolution GIFs**: Maintenance now auto-generates sequence GIFs for each species showing their progression across generations.
-   **TSV Telemetry**: All history and events are logged to flat TSV files (`results/evolution_history.tsv` and `results/events.tsv`) for easy Pandas analysis.

## Current Results

The simulation tracks the performance of all evolved agents and scenes in a centralized leaderboard.

*   **Leaderboard**: See [LEADERBOARD.md](./LEADERBOARD.md) for full rankings and summary stats. Over **190 configurations** have been evaluated!
*   **Top Performer**: Currently led by **`giraffe_default_0000`** with a positive walking score of **5.39**!
*   **Family Tree**: The leaderboard includes a Mermaid diagram visualizing the lineage and synthesis events of all saved agents!

## Interactive Controls

When running the simulation via `cli.py` (or `make run-template`), you can use the following keyboard controls in the viewer window:

-   **Spacebar**: Pause/Resume simulation.
-   **G**: Invert gravity (flips direction).
-   **+ / =**: Increase simulation speed (steps per frame).
-   **- / _**: Decrease simulation speed (steps per frame).
-   **W**: Increase wind.
-   **S**: Decrease wind.
-   **R**: Respawn all agents.

## Codebase Structure

The project uses a flat directory structure for Python files to keep it simple. Here are the most important files:

- [cli.py](./cli.py): The main entry point for running targeted simulations. Use it to load specific scene templates.
- [auto_evolve.py](./auto_evolve.py): The automated evolution runner. It handles the Genetic Algorithm and curriculum learning in the background.
- [demo.py](./demo.py): A specialized script that pulls a random selection of your best evolved agents and drops them into a chaotic environment with a giant boulder and food!
- [orchestrator.py](./orchestrator.py): The core logic engine. It combines XMLs, handles collisions, manages hunger/food, and triggers synthesis.
- [agent.py](./agent.py): Defines the `Agent` and `ConfigurableAgent` classes.
- [environment.py](./environment.py): Manages world generation (floor, sky, terrain, wind).

## Example Configurations

### Agent Config (YAML)
```yaml
agents:
- name: aegis_turtle
  type: configurable
  body:
    type: ellipsoid
    size: [0.4, 0.4, 0.1]
  limbs:
    - name: front_left
      pos: [0.25, 0.25, -0.05]
      axis: [0, 1, 0]
      range: [-20, 20]
      geom: {type: capsule, size: [0.04, 0.05]}
```

### Scene Config (YAML)
```yaml
environment:
  floor_size: [20.0, 20.0, 0.05]
  rough_terrain: true
  camera:
    pos: [0, -20, 20]
    xyaxes: [1, 0, 0, 0, 0.707, 0.707]
agents:
  - name: seer_1
    type: giraffe_default__b2dcd29b__gen20
    pos: [0.0, 0.0, 1.0]
```
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
-   **Differentiable Simulation (SHAC)**: Explore learning policies by backpropagating gradients directly through the simulator using MuJoCo MJX and contact smoothing (inspired by [differential_policies](https://github.com/saucesaft/differential_policies/)). Out of scope for now as it requires JAX.

## Project Jargon

To keep our communication precise, we use the following terminology in this project:

*   **Agent**: A physical creature simulated in MuJoCo (e.g., a specific instance of a Giraffe or Spider).
*   **Species**: A specific body structure template defined in YAML (e.g., `giraffe_default`, `scorpion_default`).
*   **Synthesis**: The event triggered when two agents collide, resulting in a new agent that inherits properties from both.
*   **Genome**: The set of parameters defining an agent's gait (frequency, phase, offsets, leg length scale).
*   **Curriculum**: The scheduled increase in environment difficulty (terrain, wind) across generations to force learning.
*   **God Mode**: The ability of the operator to inject obstacles, change gravity, or spike wind in real-time via keyboard controls to challenge the agents.

### Advanced Multiplayer Interaction
-   **Bidirectional Control**: Implement control inputs from clients to the server (e.g., applying external forces or "pushing" agents).
-   **Permissions & Roles**: Implement account-based permissions to gate access to controls (Observer vs. Puppeteer vs. Director).

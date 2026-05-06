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
4.  **Parallel Evolution**: Run multiple species in parallel to utilize multi-core systems.
    ```bash
    make parallel-evolve
    ```
5.  **Re-render Templates**: Regenerate GIFs and JPGs for all species and scenes.
    ```bash
    make rerender-all
    ```
    Or render a specific template:
    ```bash
    make render template=templates/agents/gorilla/gorilla_default.yaml output=output.gif options="--format gif"
    ```

## Overview

The goal of this project is to create an interactive environment where a physics simulation runs (powered by MuJoCo) and external players can connect via a minimal network layer to observe and interact with it. Think of it as a crowd-controlled simulation or a "Truman Show" for AI agents, where the audience can play "God" and manipulate forces, gravity, or terrain.

## System Architecture

This project implements a server-client architecture for a live-streamed, multiplayer physics simulation.

### Local Simulation & Orchestration
-   **Orchestrator**: The `orchestrator.py` module combines the generated XML from `environment.py` and `agent.py` into a single MuJoCo model.
-   **Visualization**: On macOS, the simulation is executed via the `mjpython` launcher to support the passive viewer (`launch_passive`) without thread conflicts.

### Multiplayer & Streaming (The "Puppeteer" System WIP)
*   **Simulation Server**: The `server.py` script runs the authoritative MuJoCo simulation and broadcasts the state vector to connected clients.
*   **State Streaming**: Instead of video, the server streams raw state vectors (positions, velocities, forces) via WebSockets. This minimizes bandwidth and allows clients to render locally.
*   **Clients**: The `client.py` script demonstrates connecting to the server and receiving state updates.

### Folder Structure
- `templates/agents/`: Active base templates and best evolved configurations per species.
- `templates/graveyard/`: Retired or deprecated agent templates (morphologies which underperformed or failed stability tests).
- `templates/scenes/`: Scene templates defining environment, obstacles, and active agents.
- `results/agents/`: Live results, history TSV files, and GIFs.
- `results/results_v*/`: Archived results from previous runs (Frozen).

### Results

#### Evolution Progress
<img src="results/progress.png" width="800">

#### Current Best Performer: Goliath Crawler (Score: 6.89)

> **Goliath Crawler Showcase (`goliath_crawler__9d86ac27__gen20`)**
> The Goliath Crawler currently leads the leaderboard, utilizing an ultra-stable, low-center-of-gravity gait with highly coordinated, synchronized multi-limb ground-contact cycles to maximize translation distance.
>
> ![Goliath Crawler Best Gait](templates/agents/goliath_crawler/goliath_crawler_best.gif)

#### Featured Agents

These are the primary, active species undergoing rapid optimization in active environments.

| Agent | Preview | Locomotion Design / Gait Strategy |
| :--- | :--- | :--- |
| **Khepri Beetle** | <img src="templates/agents/khepri_beetle/khepri_beetle_default.gif" width="200"> | Flat, low-slung insectoid body utilizing rapid high-frequency crawling steps. |
| **Aegis Turtle** | <img src="templates/agents/aegis_turtle/aegis_turtle_default.gif" width="200"> | Thick domed shell with wide lateral flippers producing a slow but extremely stable crawling motion. |
| **Gorilla** | <img src="templates/agents/gorilla/gorilla_default.gif" width="200"> | Quadrupedal locomotion using knuckle-walking pivots with high-mass forward lunges. |
| **Hexapod** | <img src="templates/agents/hexapod/hexapod_default.gif" width="200"> | Balanced six-legged tripod gait offering robust stability over uneven terrains. |
| **Snake** | <img src="templates/agents/snake/snake_default.gif" width="200"> | Multi-segment lateral undulation utilizing joint phase offsets for fluid slithering locomotion. |
| **Stilts Biped** | <img src="templates/agents/stilts_biped/stilts_biped_default.gif" width="200"> | Tall, slender two-legged gait leveraging long legs for massive stride lengths. |
| **Arachne Spider** | <img src="templates/agents/arachne_spider/arachne_spider_default.gif" width="200"> | Dynamic eight-legged radial sprawling gait with wide climbing and walking stability. |
| **Centipede** | <img src="templates/agents/centipede/centipede_default.gif" width="200"> | Long multi-segment structure propagating waves of leg movement for forward traction. |
| **Starfish** | <img src="templates/agents/starfish/starfish_default.gif" width="200"> | Radial five-legged sprawl utilizing crawling motions in multiple directions. |
| **Giraffe** | <img src="templates/agents/giraffe/giraffe_default.gif" width="200"> | Ultra-tall neck and leg structure achieving high translation distance per step. |
| **Goliath Crawler** | <img src="templates/agents/goliath_crawler/goliath_crawler_default.gif" width="200"> | Low-to-the-ground massive crawler utilizing high-torque synchronized leg cycles. |
| **Scorpion** | <img src="templates/agents/scorpion/scorpion_default.gif" width="200"> | Multi-legged forward sprawl with a high tail structure providing stable counterweighting. |
| **Kangaroo** | <img src="templates/agents/kangaroo/kangaroo_default.gif" width="200"> | Dual hind limbs with knee joints coupled with tail counterweighting for hopping dynamics. |
| **Crab** | <img src="templates/agents/crab/crab_default.gif" width="200"> | Lateral side-stepping crawling gait with wide leg placement for side-to-side stability. |
| **Ein Corgi** | <img src="templates/agents/ein_corgi/ein_corgi_default.gif" width="200"> | Quadrupedal low-clearance bounding gait producing rapid short-stride forward pacing. |
| **Legion Hexapod** | <img src="templates/agents/legion_hexapod/legion_hexapod_default.gif" width="200"> | Extended six-legged design featuring sprawling limbs for multi-angle load distribution. |
| **Rolling Agent** | <img src="templates/agents/rolling_agent/rolling_agent_default.gif" width="200"> | Circular geometry utilizing rotating mass-shifting dynamics to roll forward. |
| **Dragon** | <img src="templates/agents/dragon/dragon_default.gif" width="200"> | Multi-jointed elongated quadruped utilizing broad crawling sweeps for fluid motion. |
| **Chimera** | <img src="templates/agents/chimera/chimera_default.gif" width="200"> | Asymmetrical structure combining mixed joint ranges to negotiate complex terrain. |
| **Asymmetric Quadruped** | <img src="templates/agents/asymmetric_quadruped/asymmetric_quadruped_default.gif" width="200"> | Asymmetrical leg lengths optimized to exploit specific lateral gait dynamics. |


#### Archived (Graveyard) Agents

These species did not achieve stable forward locomotion, suffered from structural vulnerabilities (like high mass-to-joint ratios), or were retired in favor of superior variants. They are archived in `templates/graveyard/agents/`.

| Agent | Preview | Key Limiting Factor |
| :--- | :--- | :--- |
| **Urchin** | <img src="templates/graveyard/agents/urchin/urchin_default.gif" width="200"> | Stumpy, high-friction geometry preventing ground translation. |
| **Elephant** | <img src="templates/graveyard/agents/elephant/elephant_default.gif" width="200"> | High torque consumption resulting in rapid starvation. |
| **Frog** | <img src="templates/graveyard/agents/frog/frog_default.gif" width="200"> | Uncoordinated jumping gait leading to constant flipping. |
| **Tarantula** | <img src="templates/graveyard/agents/tarantula/tarantula_default.gif" width="200"> | Complex joint alignment causing limbs to tangle/collide. |
| **Hopper Agent** | <img src="templates/graveyard/agents/hopper_agent/hopper_agent_default.gif" width="200"> | Single-axis balance failure resulting in immediate falling. |

#### Example Scenes

| Scene | Preview | Highlight / Obstacles |
| :--- | :--- | :--- |
| **Trampoline** | <img src="templates/scenes/cym_trampoline.jpg" width="200"> | High-elasticity floor for bounce physics. |
| **Gladiator Arena** | <img src="templates/scenes/gladiator_arena.jpg" width="200"> | Enclosed ring with large rolling boulders. |
| **Chaos Colosseum** | <img src="templates/scenes/chaos_colosseum.jpg" width="200"> | Complex maze with static blocks and spheres. |
| **Neon Grid** | <img src="templates/scenes/neon_grid.jpg" width="200"> | High-contrast grid for pure movement tracking. |
| **Desert Oasis** | <img src="templates/scenes/desert_oasis.jpg" width="200"> | Uneven terrain with sand-like friction. |

#### Live Demo

**Chaotic Multi-Species Arena**
A specialized script that pulls a random selection of your best evolved agents and drops them into a chaotic environment with falling spheres, blocks, and a "Ring of Death" boundary!

<img src="results/demo/evolution.gif" width="800">


## Synthesis

The project implements a dynamic synthesis (breeding) mechanism where agents can collide and produce offspring at runtime.

### How it Works:
- **Trigger**: Synthesis is triggered by physical collision between two different agents, provided they are not on cooldown.
- **Morphology**: The child agent inherits the complete physical body structure (limbs) from ONE of the parents, chosen randomly if both are `ConfigurableAgent` instances. This ensures realistic and valid body structures.
- **Traits (Gait Parameters)**: Continuous traits are combined by averaging the parents' values and adding a small random mutation to encourage exploration:
    - **Size Scale**: Average of parents' scales ± random noise.
    - **Frequency**: Average of parents' gait frequencies ± random noise.
    - **Phase**: Average of parents' gait phases ± random noise.
    - **Phase Offsets**: Average of parents' independent phase offsets per limb ± random noise.
- **Colors**: The child's color is an average of the parents' colors, with slight random mutation.

## Simulation Mechanics & Rules

To create a more dynamic and realistic simulation, several advanced mechanics have been implemented:

### Energy & Survival
- **Energy**: Agents start with a maximum energy scaled by their size. Energy is consumed over time (hunger) and by applying torque to joints (effort).
- **Food**: Eating food restores energy and increases movement frequency by 20% (Food Excitement).
- **Death**: Agents die if they run out of energy (starvation) or remain fallen for more than a threshold time.

### Locomotion & Steering
- **Virtual Chemotaxis**: Agents compute a normalized relative position vector to the closest food target, allowing closed-loop goal seeking.
- **Asymmetric Torque Modulation**: Turning is achieved by increasing amplitude on the side contralateral to the target.
- **Fatigue**: Actuator amplitude scales linearly with agent energy to simulate exhaustion.

### Morphology & Constraints
- **Configurable Agents**: Agents are defined via YAML templates specifying body and limb hierarchy.
- **Root Body Requirement**: To prevent morphology glitches, templates must specify a `body` field for the torso, or they fallback to a default small square.
- **Limb Connection Test**: A test ensures limbs are placed within a reasonable distance of the parent body to avoid detached limbs.

## Status & Milestones

-   Foundation, Environment, Agent, and Orchestration are fully implemented and tested.
-   Visualization is implemented using `launch_passive` and `mjpython` for Mac compatibility. Interactive keyboard controls are available in `cli.py`.
-   Networking is started with state streaming via WebSockets in `server.py` and `client.py`.
-   Learning & Evolution is handled by an automated Genetic Algorithm in `auto_evolve.py`.

- **Simplified Index**: `results/index.yaml` is reduced to a clean folder-to-file-count mapping.
- **Species Censuses**: Maintenance tracks total variant counts per species (e.g., total Gorillas vs. total Snakes).
- **Smooth Playback**: Evolution GIFs are compiled at $20\text{ fps}$ ($50\text{ms}$ intervals) to make analyzing movement gaits easier on the eyes.
- **Isolated Training**: The auto-evolver evaluates agents in complete isolation to measure pure distance and survival without multi-agent interference.

### Asset Organization
-   **Centralized Rendering**: Centralized all rendering logic in `render.py` supporting GIF, JPG, and PPM formats.
-   **Asset Organization**: Reorganized `templates/agents` and `results` into species subfolders for better organization.
-   **GIF Previews**: Implemented short GIF loops for agent templates to visualize gait.
-   **Cleanup**: Removed redundant scripts and bad models to keep codebase and templates clean.

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
-   **Learning & Evolution**: Run `auto_evolve.py` to evolve agent walking parameters using a Genetic Algorithm.
-   **Automated Evolution**: Run `auto_evolve.py` to evolve multiple species sequentially and save their best configurations.
-   **Leaderboard**: Run `update_leaderboard.py` to evaluate all templates and rank them in `LEADERBOARD.md`.
-   **Per-Species Evolution GIFs**: Maintenance now auto-generates sequence GIFs for each species showing their progression across generations.
-   **TSV Telemetry**: All history and events are logged to flat TSV files (`results/evolution_history.tsv` and `results/events.tsv`) for easy Pandas analysis.

## Current Results & Strategy

The simulation tracks the performance of all evolved agents and scenes in a centralized leaderboard.

*   **Leaderboard**: See [LEADERBOARD.md](./LEADERBOARD.md) for full rankings and summary stats. Over **520 configurations** have been evaluated across **57 unique species/variants**!
*   **Top Performer**: Led by the **`goliath_crawler__9d86ac27__gen20`** template (Score: **6.89**), showcased above.
*   **Family Tree**: The leaderboard includes a visual family tree representing the breeding and lineage of saved templates.

### Evolution & Evaluation Strategy

1.  **Isolated Benchmarking**: To establish an authoritative, repeatable, and noise-free ranking, all candidates on the leaderboard are evaluated inside isolated training environments. This measures pure movement (horizontal translation distance), survival time, and limb step-detection rewards without multi-agent interference, physical blocking, or food-competition starvation.
2.  **Genetic Algorithm**: Evolution runs are automated via `auto_evolve.py` using a hill-climbing/genetic approach to optimize joint frequencies, phases, leg length scale, and phase offsets.
3.  **Synthesis & Co-Evolution Status**:
    *   **Synthesis Mechanics**: Two agents of the same or different species can collide in real-time runs to trigger breeding. The offspring inherits the morphology (limbs) from one parent, with an averaged and slightly mutated gait parameter genome (frequencies, phase, colors).
    *   **Co-Evolution Status**: Although synthesis is fully active and playable during live interactive sessions and multiplayer demos, **it is not yet utilized to co-evolve or determine the authoritative leaderboard rankings**. Rankings are strictly established via isolated single-agent distance evaluations.

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

- [cli.py](./cli.py): CLI for selecting and launching simulation templates, and the main entry point for targeted simulations.
- [auto_evolve.py](./auto_evolve.py): The automated evolution runner. It handles the Genetic Algorithm and curriculum learning in the background.
- [demo.py](./demo.py): A specialized script that pulls a random selection of your best evolved agents and drops them into a chaotic environment with a giant boulder and food!
- [orchestrator.py](./orchestrator.py): The core logic engine. It combines XMLs, handles collisions, manages hunger/food, and triggers synthesis.
- [orchestrator_test.py](./orchestrator_test.py): Unit tests for the orchestrator module.
- [agent.py](./agent.py): Defines the `Agent` and `ConfigurableAgent` classes.
- [environment.py](./environment.py): Manages world generation (floor, sky, terrain, wind).
- [simulate_visual.py](./simulate_visual.py): Runs the simulation with a visual window and runtime interactions.
- [server.py](./server.py): WebSocket server for streaming simulation state.
- [client.py](./client.py): WebSocket client for receiving state updates.
- [templates/](./templates/): Folder containing YAML simulation templates (agents and scenes).
- [render.py](./render.py): General rendering utility for agents and scenes (GIF, JPG, etc.).
- [Makefile](./Makefile): Manages setup, formatting, and tests.
- [DEV_LOG.md](./DEV_LOG.md): Log of notable contributions and milestones.

## Example Configurations

This section explains how the agent YAML files define a creature, the mechanics of how they are compiled and actuated, and what each parameter means.

---

### Scene Configuration (YAML)

Scene configurations reside under `templates/scenes/*.yaml`. They define the simulation environment, global physics, obstacles, and starting agent spawns.

```yaml
environment:
  gravity: [0.0, 0.0, -9.81]
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
---

### Agent Configuration (YAML)

Agent configurations reside under `templates/agents/[species]/*.yaml`. They allow abstract morphology design without writing Python code, powered by the `ConfigurableAgent` class in `agent.py`.

Here is an example configuration showing a complex hybrid biped structure (`kangaroo_default` template) with hierarchical knee/calf joints:

```yaml
agents:
  - name: kangaroo
    type: configurable
    body:
      type: box
      size: [0.2, 0.15, 0.4]
      mass: 8.0
    limbs:
      - name: left_leg
        pos: [0.0, 0.12, -0.1]
        axis: [0, 1, 0]
        range: [-30, 30]
        geom: {type: capsule, size: [0.05, 0.2]}
        child:
          name: left_calf
          pos: [0.0, 0.0, -0.2]
          axis: [0, 1, 0]
          range: [0, 90]
          geom: {type: capsule, size: [0.04, 0.2]}
      - name: right_leg
        pos: [0.0, -0.12, -0.1]
        axis: [0, 1, 0]
        range: [-30, 30]
        geom: {type: capsule, size: [0.05, 0.2]}
        child:
          name: right_calf
          pos: [0.0, 0.0, -0.2]
          axis: [0, 1, 0]
          range: [0, 90]
          geom: {type: capsule, size: [0.04, 0.2]}
      - name: tail
        pos: [-0.2, 0.0, -0.1]
        axis: [0, 1, 0]
        range: [-30, 30]
        geom: {type: capsule, size: [0.06, 0.4]}
```

#### Parameter Schema

- **`agents`**: A list of agent definition blocks in the YAML template.
- **`name`**: Unique identifier or species name of the agent (e.g., `kangaroo`, `aegis_turtle`).
- **`type`**: Must be set to `configurable` to be loaded as a `ConfigurableAgent`.
- **`body`**: Defines the primary torso/root body of the creature.
  - **`type`**: MuJoCo geometry type (e.g., `box`, `capsule`, `sphere`, `ellipsoid`).
  - **`size`**: A 3-element list of floats `[x, y, z]` scaling the body dimensions in MuJoCo.
  - **`mass`** *(optional)*: Explicit mass of the torso.
- **`limbs`**: A list of main limbs attached directly to the `body`.
  - **`name`**: Unique identifier for the limb (e.g., `left_leg`, `front_right`).
  - **`pos`**: A 3-element offset `[x, y, z]` from the center of the parent `body` to place the limb attachment point.
  - **`axis`**: Rotation axis for the joint (e.g., `[0, 1, 0]` is a hinge joint rotating around the Y-axis).
  - **`range`**: Hinge rotation limits in degrees `[min, max]`.
  - **`geom`**: Object defining physical visual geometry:
    - **`type`**: Geometry type (e.g., `capsule`).
    - **`size`**: Scale dimensions `[radius, length]`.
  - **`child`** *(optional)*: Defines a hierarchical segment/joint (e.g. knee/calf) attached to the end of the parent limb.
    - Includes the same keys as the parent limb (`name`, `pos`, `axis`, `range`, `geom`), placed relative to the parent limb's frame.

---

### Mechanics

When the `Orchestrator` loads a `ConfigurableAgent`:
1. **XML Model Generation**: `ConfigurableAgent.generate_xml()` parses the YAML configuration and constructs the MJCF (MuJoCo XML) structure representing body geometries, freejoints, hinge joints, rangefinders (`rangefinder`), and motor actuators (`<motor>`) dynamically.
2. **Sine Wave Actuation Policy**: At each physics step, the agent's `act()` method applies joint torques to each motor using a sine wave gait controller:
   $$\text{ctrl} = \text{amplitude} \times \sin(\text{time} \times \text{frequency} + \text{phase} + \text{phase\_offset}) + \text{bias}$$
   - **Frequency & Phase**: Global parameters controlling walking speed and starting wave position.
   - **Phase Offsets**: Per-actuator offsets automatically spaced at $2\pi / N$ to ensure coordinated wave-like locomotion across $N$ limbs.
   - **Amplitude**: Gait strength, scaled by current agent energy to simulate muscle fatigue as energy depletes.
3. **Chemotaxis & Steering**: If food (red spheres) is present, the agent computes a relative position vector to the food. Using the cross product between the agent's forward heading (derived from its orientation quaternion) and the food vector, the agent determines whether the food is to the left or right. It then dynamically adds `steering_weight` to the limb amplitudes on the contralateral side to steer the body toward the food.
4. **Obstacle Avoidance**: The agent is equipped with a front-facing rangefinder sensor (`rangefinder`). If an obstacle is detected within 1.0 meter, a positive `bias` is applied to push the legs forward/backward to backup or pivot.

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

## Terminology

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

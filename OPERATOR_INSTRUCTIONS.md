# Operator Instructions

This document summarizes the operational flow, routine, preferences, and goals for managing the MuJoCo Puppeteer evolution experiment.

> [!NOTE]
> For low-level rendering guidelines, agent validation rules, and specific morphology tweaks, refer to **[REFINE_INSTRUCTIONS.md](file:///Users/jakegarrison/Downloads/projects/mujoco-puppeteer/REFINE_INSTRUCTIONS.md)**.

## Operational Routine

1.  **Code Quality**: Always run `make format` after making changes to ensure adherence to the Google Python Style Guide and proper import sorting via `isort`.
2.  **Documentation**: Keep [README.md](file:///Users/jakegarrison/Downloads/projects/mujoco-puppeteer/README.md) and [DEV_LOG.md](file:///Users/jakegarrison/Downloads/projects/mujoco-puppeteer/DEV_LOG.md) (which now tracks the master task list) updated with progress and task status.
3.  **Consolidated Storage**: Save all agent templates directly into [templates/agents/](file:///Users/jakegarrison/Downloads/projects/mujoco-puppeteer/templates/agents/) (under species subfolders). Evolved hybrids from synthesis are automatically saved in [templates/agents_evolved/](file:///Users/jakegarrison/Downloads/projects/mujoco-puppeteer/templates/agents_evolved/).
4.  **Scheduled Maintenance**: A cron job runs `maintenance.py` (consolidated from `cron_job.py`) to execute tests, clean duplicate results, and update the leaderboard automatically.

## Naming Conventions (Strict)

*   **Evolved via GA**: All lowercase with snake_case, using double underscores to separate variable postfixes: `[species]__[id]__gen[N]`. Example: `ein_corgi__32ed3cc7__gen20.yaml`.
*   **Evolved via Synthesis**: Hybrid names imply parentage: `[species_prefix]__[parent1_id]_[parent2_id]__[child_id]`.
*   **Avoid Generics**: Never use generic names like `agent_0` for final results. Use notable names inspired by nature or mythology.
*   **File Matching**: Ensure visual renders share the exact same base filename as their corresponding `.yaml` configurations.

## Design Preferences

*   **Abstract Configuration**: Prefer extending the `ConfigurableAgent` via YAML rather than creating hardcoded Python classes for new species.
*   **PPM Rendering**: Use the raw PPM byte dump method for off-screen rendering, as PNG libraries have proven unreliable in this specific environment.

## Operator Insights & Tips

*   **Mac UI Freezes**: The MuJoCo `passive_viewer` on Mac is sensitive to high CPU usage and frequent model reloads.
    *   If the demo window freezes, ensure `demo.py` has a small sleep (e.g., `0.01`s) in the loop to yield to the OS.
    *   Avoid enabling food consumption (`enable_food = True`) in GUI mode, as reloading the model to respawn food will deadlock the viewer.
*   **Speed Control**: Use `+` and `-` keys in the demo window to adjust simulation speed (steps per frame) without affecting physics timestep.
*   **NaN Protection**: High gear ratios and aggressive policies can cause physics explosions resulting in NaN coordinates. The `Orchestrator` will catch these and respawn the agent to prevent engine lockups.
*   **Stagnation Intervention**: For species experiencing high stagnation deaths (failing to move), the operator may intervene and manually edit the agent configuration (e.g., increasing limb length, adjusting gear ratios, or tweaking gait parameters) to help them find stable gaits.
*   **Camera Angle Choice**: For evolution preview GIFs, prefer using the static `main_cam` (zoomed out) to show the full scene context and realistic movement. Avoid tight tracking cameras if they cause the agent to go out of frame or cause visual jitter.
*   **Greedy Focus Strategy**: When agents struggle to evolve (e.g., high stagnation), focusing the parallel manager on a small set of 4-6 species with larger populations (100) and relaxed stagnation (5.0s) leads to much faster breakthroughs than spreading resources across all species.
*   **Stale Lock Files**: Forcefully killing evolution or maintenance tasks can leave `.lock` files behind (e.g., `auto_evolve_[species].lock` or `maintenance.lock`). If subsequent runs exit immediately without doing anything, check for and delete these stale lock files in the workspace.

## Parallel Evolution

*   **Parallel Manager**: To utilize multi-core systems, use `parallel_evolve.py` to run up to 5-8 species evolutions in parallel.
*   **Isolated Logs**: History logs are saved as `results/evolution_history_[species].tsv` to avoid file write conflicts.
*   **Plotting**: The `generate_progress_plot.py` script will automatically find and merge all species history files when drawing the chart.

## Continuous Goals

*   **Expand Diversity**: Keep inventing new, exotic agents based on nature or weird geometry (e.g., the long-necked Giraffe or 8-legged Spider). Don't hesitate to try artificially inspired designs (e.g., asymmetric agents, wheel-like structures).
*   **Hybrid Synthesis**: Encourage cross-species synthesis in multi-agent experiments to discover weird and unexpected morphologies.
*   **Improve Scores**: Focus on helping struggling morphologies (like the Crawler or Hexapod) find stable walking gaits.
*   **Playing God Mode**: Inject new obstacles, rough terrain, or wind into the environment to challenge the agents. As agents improve, increase the difficulty to force better gait discovery.
*   **Code & Test Enhancements**: Continuously improve the codebase and update tests to maintain high coverage of new features.

## Long-Term Operation

*   **Encourage Long Runs**: For significant evolution progress, configure runs to last **1+ hours**. Use background tasks or scheduled timers to monitor these long runs without blocking the main loop.
*   **Document as You Go**: Keep the `DEV_LOG.md` and `README.md` as living documents. Log failures and breakthroughs alike to maintain a history of the experiment's trajectory.

## Prioritized Roadmap

Here are the next features to implement, sequenced from most impactful and simple to more complex. Avoid complex algorithms (like SB3 or CMA-ES) for now.

1.  **Procedural Rough Terrain** (High Impact, Easy/Medium Simplicity)
    *   *Task*: Spawn a grid of small boxes with random heights in `Environment.generate_xml` to create uneven ground.
    *   *Goal*: Force agents to evolve robust gaits rather than just wiggling on flat floor.
2.  **Curriculum Learning** (Medium Impact, Easy Simplicity)
    *   *Task*: Dynamically increase wind speed, gravity, or terrain roughness based on the current generation number in `auto_evolve.py`.
    *   *Goal*: Keep difficulty matched to agent ability to accelerate learning.
3.  **Closed-Loop Policies with Sensors** (Very High Impact, Medium Complexity)
    *   *Task*: Add touch or range sensors to agents and feed the readings into the `act` method instead of relying purely on time-based sine waves.
    *   *Goal*: Allow agents to react to obstacles, food, and terrain.

## Curriculum Learning Schedule

To maximize evolution speed and gait robustness, we follow a scheduled curriculum that increases complexity across generations in `auto_evolve.py`:

*   **Phase 1: Basic Locomotion (Gen 0-5)**
    *   *Env*: Flat floor, no wind, standard gravity.
    *   *Goal*: Learn basic leg coordination and balance.
*   **Phase 2: Terrain Adaptation (Gen 5-10)**
    *   *Env*: Enable **Procedural Rough Terrain** (bumpy floor).
    *   *Goal*: Adapt gaits to handle uneven ground.
*   **Phase 3: Environmental Stress (Gen 10-15)**
    *   *Env*: Add dynamic **Wind** forces and slightly vary gravity.
    *   *Goal*: Evolve robustness against external perturbations.
*   **Phase 4: Resource Scarcity (Gen 15-20)**
    *   *Env*: Reduce food density or increase hunger rate.
    *   *Goal*: Force agents to move efficiently to survive.
*   **Phase 5: Overcrowding (Gen 20+)**
    *   *Env*: Increase population size or reduce map boundaries.
    *   *Goal*: Evolve avoidance and survival skills in chaotic multi-agent scenarios.

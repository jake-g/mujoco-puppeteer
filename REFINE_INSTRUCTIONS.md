# Refinement Instructions

This document tracks instructions and best practices for refining agents and environments in `mujoco-puppeteer`, focusing on fixing bugs and optimizing for autonomous sampling.

## Rendering Guidelines

### GIFs (Template Preview)
- **Environment**: Always use zero gravity (`[0.0, 0.0, 0.0]`) to prevent agents from falling or flying away during the preview.
- **Floor**: Hide the floor grid to focus on the agent. Use the tiny floor size hack (`[0.001, 0.001, 0.001]`).
- **Position**: Force the agent's root body position to `[0.0, 0.0, 1.0]` at every step to keep it centered in frame while wiggling.
- **Camera**: Use static `main_cam` positioned at `[0.0, -2.5 * scale, 3.5 * scale]` looking at `[0, 0, 1]`.
- **Speed**: Render at 20 fps (`duration=50ms`).

### JPEGs (Static Preview)
- **Environment**: Use the default environment with the floor grid visible.
- **Position**: Let the agent settle naturally for 10 steps.
- **Camera**: Same as GIFs.

## Agent Validation

### Geometry Check
- Check for limb detachment! Limbs should not be placed too far from their parent bodies.
- Use `test_limb_connections` in `templates_test.py` to verify that limb distance is within `3.0 * max_torso_dimension`.

### Morphology Definition
- **Root Body**: Always specify a `body` section at the root of the agent config in YAML to define the torso size and shape.
- If missing, `ConfigurableAgent` falls back to a default small square (`[0.2, 0.2, 0.05]`), which might disconnect limbs placed further out!
- Do NOT put the torso dimensions in the `limbs` list as a limb named "torso", as it will be ignored for the root body!

## Evolution & Optimization

### Simulation Parameters
- Based on precision experiments (Subagent `f1f3bd6c`):
    - Using a larger timestep (e.g., `0.005` or `0.010`) can speed up simulation by 5x without hurting performance.
    - Lower gravity (e.g., `-5.0` or `-2.0`) can help certain species (like Starfish) perform better and reach further distances sooner.

### Leaderboard Metrics
- **Food Eaten**: Tracked per agent and displayed on the leaderboard. Touching food gives energy and increases frequency.
- **Breeding Events**: Tracked per agent (syntheses count) and displayed on the leaderboard.
- **Stagnation Death**: Agents must move at least `0.1` meters within `3.0` seconds or they are killed, preventing static survival strategies.

### Genome Expansion
- **Steering Weight**: Added to the agent genome. It determines how strongly the agent modulates its amplitude to steer towards food. This allows agents to evolve better food-seeking behavior!

### Refinement Strategy
- **Stick to Identity**: When an agent fails to move or has issues, refine its morphology but stick to its expected shape and identity (e.g., Elephant should remain big).
- **Greedy Focus**: Focus on a few agents at a time to improve distance and time to max distance, then switch to another set for consistent results.
- **Rotation**: Shift to a different set of 8 agents after ~20 minutes or upon reaching a milestone to ensure all agents eventually improve.
- **Automated Reflection**: Every 10 minutes, a recurring task prompts the agent to read task logs, analyze agent performance, update `DEV_LOG.md` with new findings, and suggest/invent new agent configs.

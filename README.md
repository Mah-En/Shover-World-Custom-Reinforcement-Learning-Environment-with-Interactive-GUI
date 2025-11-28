# Shover-World

Shover-World is a small but expressive grid-based environment for experimenting with decision-making, planning, and resource management in artificial intelligence. It models an agent that moves on a 2D grid, pushes chains of boxes, manages limited stamina, and performs **special actions** that can reshape the world (creating barriers and lava). A Pygame-based GUI is included for interactive play and visualization. :contentReference[oaicite:0]{index=0}

---


## Performance visualization 
![video of it's interaction in a small grid](demo.mp4)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Environment Concepts](#environment-concepts)
  - [Grid and Objects](#grid-and-objects)
  - [Agent and Stamina](#agent-and-stamina)
  - [Movement and Chain Push](#movement-and-chain-push)
  - [Perfect Squares](#perfect-squares)
  - [Special Actions](#special-actions)
- [Environment API](#environment-api)
  - [Gym-style Interface](#gym-style-interface)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Step Logic](#step-logic)
- [Grid Representation and Map Loading](#grid-representation-and-map-loading)
- [GUI (Pygame) Interface](#gui-pygame-interface)
  - [HUD / On-screen Info](#hud--on-screen-info)
  - [Keyboard Controls](#keyboard-controls)
  - [Mouse Controls](#mouse-controls)
- [Experiments and Observations](#experiments-and-observations)
- [Planned / Possible Future Work](#planned--possible-future-work)
- [Citation](#citation)

---

## Overview

Shover-World was designed as part of an Artificial Intelligence course (Fall 2025) to explore how a simple grid world can become complex and interesting by combining:

- **Spatial structure** (boxes, barriers, lava on a grid),
- **Resource constraints** (stamina),
- **Non-trivial interaction rules** (chain pushing),
- **Pattern recognition in the environment** (perfect squares),
- **Map-transforming actions** (Barrier Maker and Hellify),
- **Real-time visualization** via a Pygame GUI.

Although compact, the environment is compatible with the **OpenAI Gym-style API**, making it suitable both for **manual play** and for future **reinforcement learning experiments**.

---

## Key Features

-  **Gym-style environment** (`reset`, `step`, `close`, observation/action spaces).
-  **Chain pushing** of multiple boxes with variable stamina cost.
-  **Limited stamina** with extra costs for pushing and refunds from lava.
-  **Perfect square detection** (n × n box formations with clean borders).
-  **Two special actions** that transform perfect squares:
  - **Barrier Maker**: turns boxes into barriers and gives stamina.
  - **Hellify**: digs a lava pit inside a square (for n ≥ 3).
-  **Aging and auto-dissolving squares**, introducing time-based strategy.
-  **Pygame GUI** for visualization, debugging, and manual play.
-  **Random or file-based map generation** with robust validation.

---

## Environment Concepts

### Grid and Objects

The environment is an `nrows × ncols` grid stored as a NumPy array of integers:

- **Empty cell**: `0`
- **Box**: values in `[1, 10]` (sub-range reserved for boxes)
- **Barrier / wall**: `100` (solid, immovable, indestructible)
- **Lava**: `-100` (destroys boxes and may refund stamina)

The **agent’s position** is stored separately as a `(row, col)` tuple:

```python
agent_pos = (r, c)
````

This keeps the grid purely object-based and makes movement logic simpler.

### Agent and Stamina

The agent has a finite **stamina** value that:

* **Decreases** with every action (even invalid ones),
* **Decreases more** when pushing chains of boxes,
* Can **increase** (refund) when pushing boxes into lava,
* Reaches **zero** at the end of an episode (the agent “exhausts” itself).

The environment also tracks:

* `timestep`: how many steps have elapsed,
* `last_pushed_head`: information about the last push direction/position,
* A **square registry**: list of detected perfect squares and their ages.

### Movement and Chain Push

Movement actions (up, right, down, left) are defined on the grid:

* If the **target cell is empty**, the agent moves there; cost is `1` stamina.
* If the **target cell has a box**, the agent attempts a **chain push**.

**Chain pushing:**

1. Collect all consecutive boxes in the chosen direction, forming a chain of length `k`.
2. Inspect the cell after the last box:

   * **Empty** → push succeeds; all boxes shift by one cell.
   * **Lava** → push succeeds; last box moves into lava and is destroyed.
   * **Barrier or another box** → push fails; nothing moves.

Invalid movement or push attempts **still cost 1 stamina**, but leave the agent in place.

#### Stamina Cost for Pushes

The cost for pushing a chain of `k` boxes uses a force-based model:

* **New push direction** (different from last push):
  [
  \text{PushCost} = F_0 + k \cdot F_u
  ]
* **Continuing in the same direction**:
  [
  \text{PushCost} = k \cdot F_u
  ]

Where:

* `F0` = initial force cost,
* `Fu` = per-box cost.

This models that **starting** to push is harder than **continuing** in the same direction.

#### Stamina Refund from Lava

If the last box in a push chain is moved into a lava cell:

* The box is destroyed.
* The agent receives a **stamina refund**:
  [
  \text{Refund} = F_0
  ]

This creates a trade-off:

* Keep boxes around for future patterns and pushes, or
* Sacrifice them into lava for extra stamina.

### Perfect Squares

Perfect squares are central to the environment and special actions.

A **perfect square** is defined as:

* An `n × n` block of **box cells** (`n ≥ 2`),
* With **no boxes** in the one-cell wide border around it.

Formally:

1. For all cells inside the candidate block:
   [
   \forall (i, j) \in \text{block},; \text{grid}[i, j] \in [1, 10]
   ]
2. For all cells in the surrounding border:

   * No cell contains a box value.

The environment **scans for perfect squares every step** using a `find_perfect_squares()` routine.

#### Square Registry and Aging

Squares are tracked in a registry with entries:

* `n` – size of the square,
* `(r, c)` – top-left coordinate,
* `age` – how many steps it has existed.

At every step:

1. Squares are re-detected.
2. Existing squares have their `age` incremented.
3. New squares are added with `age = 0`.
4. Disappeared squares are removed from the registry.

When a square’s age exceeds a threshold:

[
\text{age} \ge \text{perf_sq_initial_age}
]

…the square **dissolves** automatically: all its cells become empty. This prevents the grid from being filled permanently and adds a timing element: **use squares before they expire**.

### Special Actions

There are two special actions that can only be applied when at least one perfect square exists.

#### Barrier Maker (Action 5)

Turns a chosen perfect square into a solid barrier block:

1. Detect all current perfect squares.
2. If none exist → action is invalid (still costs stamina).
3. Otherwise, **choose**:

   * The **smallest** square by size `n`,
   * If tied, the **oldest** one (largest `age`).
4. Convert every cell in the `n × n` area to a barrier value (`100`).
5. Add a stamina reward:
   [
   \Delta \text{stamina} = n^2
   ]

Use cases:

* Blocking corridors or regions,
* Protecting areas,
* Strategic stamina gain.

Because barriers are **permanent and immovable**, poor use can block the agent’s own paths.

#### Hellify (Action 6)

Converts the **interior** of a square into a lava pit (requires `n > 2`):

* Border cells of the square become **empty** (`0`),
* Inner cells become **lava** (`-100`),
* Any boxes in that interior are immediately destroyed.

Example (3 × 3):

From:

[
\begin{bmatrix}
10 & 10 & 10 \
10 & 10 & 10 \
10 & 10 & 10
\end{bmatrix}
]

To:

[
\begin{bmatrix}
0 & 0 & 0 \
0 & -100 & 0 \
0 & 0 & 0
\end{bmatrix}
]

This allows the agent to clear large groups of boxes at once, potentially creating dynamic lava zones that can later be used for stamina refunds via pushes.

---

## Environment API

### Gym-style Interface

The environment is implemented as a Python class (e.g. `ShoverWorldEnv`) with a **Gym-like API**:

* `reset()` → starts a new episode and returns initial observation.
* `step(action)` → applies one action and returns `(obs, reward, done, info)`.
* `close()` → cleans up any resources (e.g. GUI windows).
* `observation_space` and `action_space` are defined using Gym’s `spaces`.

Example usage (adjust import path to your project layout):

```python
import gym
# from your_module import ShoverWorldEnv  # adjust this line to your actual module name

env = ShoverWorldEnv(
    grid_size=(10, 10),
    n_boxes=10,
    n_barriers=5,
    n_lava=3,
    initial_stamina=50,
    # other configuration options...
)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # random agent
    obs, reward, done, info = env.step(action)

env.close()
```

### Action Space

The action space is represented as a Gym `spaces.Tuple`:

```python
spaces.Tuple((position, action_type))
```

* `position` – a grid position (included to satisfy assignment requirements; not used for movement logic).
* `action_type` – an integer encoding the type of action:

  | Action Type | Description        |
  | ----------- | ------------------ |
  | 0           | No-op (do nothing) |
  | 1           | Move Up            |
  | 2           | Move Right         |
  | 3           | Move Down          |
  | 4           | Move Left          |
  | 5           | Barrier Maker      |
  | 6           | Hellify            |

### Observation Space

The observation returned from `step()` and `reset()` is a **dictionary** containing:

* `grid`: the current grid as a NumPy `int32` array of shape `(nrows, ncols)`,
* `agent_pos`: the agent’s `(row, col)` position,
* `stamina`: current stamina value,
* `last_position`: last selected position (from the action tuple),
* `last_action`: last action type.

Everything is encoded in Gym-compatible formats so standard RL libraries can integrate without modification.

### Step Logic

The `step(action)` method is the core of the environment and processes, in this order:

1. Read and validate the action tuple.
2. Try to move the agent or perform a chain push.
3. Apply stamina costs for movement or pushing (or invalid action).
4. Handle box destruction in lava and stamina refunds.
5. Detect all current perfect squares.
6. Update the square registry (increase ages, add new squares, remove vanished ones).
7. Dissolve squares that have exceeded their maximum age.
8. Apply special actions (Barrier Maker or Hellify) if chosen.
9. Construct the next observation and compute the reward.
10. Return `(observation, reward, done, info)`.

The `info` dict can include, for example:

* `valid_action`: whether the action was valid or not,
* `n_boxes_pushed`: number of boxes pushed this step,
* `n_boxes_destroyed`: number of boxes destroyed (e.g. by lava),
* `perfect_squares`: list of currently detected squares and their properties.

---

## Grid Representation and Map Loading

The environment supports both **file-based** and **random** map creation.

### Internal Representation

* Grid stored as a NumPy array of integers.
* Agent position stored separately.
* Object types encoded as simple numeric codes for efficiency and simplicity.

### Map File Formats

Two supported formats:

1. **Format A – numeric grid**

   * Each cell is directly encoded using the internal numeric values (e.g. `0`, `10`, `100`, `-100`, etc.).
   * Closely matches the environment’s internal representation.
   * Suitable for programmatic generation.

2. **Format B – symbolic grid**

   * Uses simple symbols (easier to read and edit manually).
   * Environmental loader translates symbols to the internal numeric codes.

The loader includes basic **validation**:

* Correct number of columns per row,
* Valid characters / values,
* At least one valid starting position for the agent.

### Random Map Generation

If no map file is provided:

* The agent is placed at a random free position.
* A specified number of boxes, barriers, and lava cells are placed randomly.
* This is useful for:

  * Quick experiments,
  * Testing robustness to different layouts,
  * Making gameplay more varied in the GUI.

---

## GUI (Pygame) Interface

Although the environment can run headless (without graphics), the **Pygame-based GUI** is a major part of the project and extremely useful for:

* Debugging,
* Understanding the dynamics,
* Demonstrations,
* Manual experiments on strategies.

### Rendering

* Pygame is used to open a window and render:

  * The grid (each cell as a colored rectangle, based on content),
  * The agent,
  * HUD information (see below).
* The **cell size** is chosen so the whole grid fits within the window:

  [
  \text{cell_size} = \frac{\text{window_size}}{\max(nrows, ncols)}
  ]

This allows arbitrary grid sizes to be visualized.

### HUD / On-screen Info

Above (or around) the grid, the GUI displays:

* **Timestep** (current step number),
* **Stamina** (remaining),
* **Number of boxes remaining**,
* **Number of boxes destroyed**,
* **Validity of the last action** (valid/invalid),
* **Boxes destroyed in the latest push**.

All of this information is obtained from the environment’s `info` dictionary after each `step()`.

### Keyboard Controls

Default key bindings:

| Key         | Action                                |
| ----------- | ------------------------------------- |
| `W` / Up    | Move Up (Action 1)                    |
| `D` / Right | Move Right (Action 2)                 |
| `S` / Down  | Move Down (Action 3)                  |
| `A` / Left  | Move Left (Action 4)                  |
| `B`         | Barrier Maker (Action 5)              |
| `H`         | Hellify (Action 6)                    |
| `R`         | Reset environment (start new episode) |
| `Q`         | Quit program                          |

Each key press:

1. Is mapped to an action tuple,
2. Passed to `env.step()`,
3. The screen is re-rendered with the new state.

### Mouse Controls

The GUI also supports mouse input:

* Clicking on a grid cell computes the corresponding coordinates `(r, c)`.
* The agent can then **jump directly to the clicked position**, as long as the path is not blocked by barriers (depending on implementation).
* This feature is convenient for:

  * Quickly exploring the map,
  * Setting up specific scenarios,
  * Rapid testing of pushing and special actions.

### Main Event Loop

The GUI runs an event loop:

1. Poll keyboard and mouse events.
2. Translate these into actions (or GUI operations like reset/quit).
3. Call `env.step()` when appropriate.
4. Draw the grid and HUD.
5. Update the display.

This loop runs until the user quits the program.

---

## Experiments and Observations

Several experiments were conducted:

1. **Random agent**:

   * At each step, a random action is chosen.
   * Typical behavior:

     * Repeats invalid moves (wasting stamina).
     * Attempts to push boxes when pushes are impossible.
     * Rarely constructs perfect squares accidentally.
     * Rarely uses lava meaningfully.
   * Most episodes end when stamina reaches zero.
   * Despite poor performance, this agent is useful to:

     * Stress-test the environment,
     * Ensure steps are robust and not crash-prone,
     * Verify that chain pushing, stamina updates, and lava interactions behave correctly.

2. **Manual play via GUI**:

   * A human player can intentionally:

     * Build perfect squares,
     * Use Barrier Maker and Hellify at specific times,
     * Create or avoid lava regions,
     * Test long chains and edge cases.
   * Observations:

     * Chain pushing is intuitive and visually clear.
     * Perfect squares are detected reliably once formed.
     * Square ages increment each step and dissolve correctly after the configured threshold.
     * Special actions significantly alter the map and require careful planning.

These experiments demonstrate that even with simple rules, Shover-World supports nontrivial planning and strategy.

---

## Planned / Possible Future Work

Some natural extensions and ideas for future work include:

* **Reinforcement learning integration**:

  * Training a learning agent (e.g., DQN, PPO) to maximize long-term rewards given stamina and map structure.
* **Goal definition**:

  * Adding explicit goals such as reaching a target cell, maximizing boxes destroyed, or maximizing surviving boxes.
* **Additional object types**:

  * Moving enemies, collectible items, doors, keys, etc.
* **More special actions**:

  * Map transformations beyond squares (e.g., line clear, region flip).
* **Advanced GUI features**:

  * Replay saving and loading,
  * Step-by-step playback,
  * Overlays for perfect squares and their ages.
* **Configurable reward functions**:

  * Rewarding specific behaviors (e.g., building squares, efficient stamina use).

---

## Citation

If you use this environment in academic work or reports, you can cite it based on:

> Mahla Entezari, *Shover World – Millstone 1 – Environment and GUI*, Artificial Intelligence course, Shahid Beheshti University, Fall 2025. 

---


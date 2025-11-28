
# ShoverWorld Environment – README
# Mahla Entezari - 401222017
## Artificial Intelligence course

This document explains the structure and behavior of the **`ShoverWorldEnv`** environment.
The purpose of this environment is to simulate the Shover-World puzzle described in the course assignment, using a Gym-compatible API.

The environment supports:

* a grid-based world with boxes, barriers, and lava
* chain pushing mechanics
* stamina-based action costs
* detection and transformation of “perfect square” box patterns
* optional map loading
* a standard Gym observation + action interface

---

# 1. Overview

ShoverWorldEnv is a subclass of gym.Env.
It implements a grid world where an agent can:

* Move in 4 directions
* Push chains of boxes
* Create barriers (action 5)
* Apply the “Hellify” transformation (action 6)
* Interact with lava, which destroys boxes
* Earn/lose stamina as part of the cost mechanics

The environment terminates when:

1. The agent runs out of stamina, OR
2. There are no boxes remaining, OR
3. The episode reaches max timestep.

---

# 2. Grid Encoding

Each cell in the grid is an integer:

| Cell Type | Meaning        | Value                         |
| --------- | -------------- | ----------------------------- |
| Empty     | Free cell      | `0`                           |
| Lava      | Destroys boxes | `-100`                        |
| Box       | Movable block  | `1–10` (actual default: `10`) |
| Barrier   | Immovable      | `100`                         |

The grid shape is n_rows × n_cols (default 8×8).

The agent is not encoded inside the grid. Its position is stored in self.agent_pos.

---

# 3. Action Space

The action space is a tuple:


((row, col), action_type)

* (row, col) = the *targeted position* (mainly logged for observations)
* action_type in {0…6}

| Action | Meaning         |
| ------ | --------------- |
| 0      | No-op / invalid |
| 1      | Move Up         |
| 2      | Move Right      |
| 3      | Move Down       |
| 4      | Move Left       |
| 5      | Barrier Maker   |
| 6      | Hellify         |

### Important Note

Even though the action contains (row, col), push behavior is determined only by action_type.
The movement always happens relative to the agent's current position.

---

# 4. Observation Space

The environment returns a dictionary:

```
{
    "grid": np.ndarray (int32),
    "agent": [row, col],
    "stamina": float32,
    "previous_selected_position": [row, col],
    "previous_action": int,
}
```


This is implemented using a Gym spaces.Dict, so it works naturally with RL agents.

---

# 5. Stamina & Action Costs

Every step has:

### Baseline cost


stamina -= 1


### Push cost

If a chain of k boxes is pushed:

* If stationary
  (i.e., the same “head box + direction” as previous step)

  
  cost = initial_force + unit_force * k

* If continuing the same push direction

  cost = unit_force * k

### Lava refund

Boxes pushed into lava give:

stamina += initial_force  (per destroyed box)

There is also an optional **positive reward** for lava destruction.

---

# 6. Chain Push Mechanics

When the agent moves into a direction:

1. If the next cell is **empty**, the agent moves.
2. If the next cell is a **box**, a chain push is attempted:

   * Find all consecutive boxes (k long)
   * Check the cell after them:

     * If barrier/box/out-of-bounds → **invalid**
     * If empty → push
     * If lava → push + destruction

Boxes are shifted from back to front to avoid overwriting.

Agent steps into the position of the first box.

---

# 7. Perfect Square Detection

Every step, the environment searches the grid for **n×n blocks of contiguous boxes** such that:

* All n² cells are boxes
* The 1-cell border around the square contains **no boxes**

Each detected square has an internal "age" counter that increments every step.

If age >= perf_sq_initial_age default (10), the square is **dissolved**, meaning all its cells become empty.

---

# 8. Special Actions

## 8.1 Barrier Maker (action 5)

* Finds all perfect squares with n ≥ 2
* Selects the smallest-n, oldest one
* Converts the whole n×n block into **barrier cells**
* Adds n² stamina

## 8.2 Hellify (action 6)

* Finds all perfect squares with n > 2
* Converts the **border** to empty
* Converts the **interior** to lava
* Counts interior boxes as destroyed

---

# 9. Map Loading

The environment supports loading maps from text files in two formats:

### Format A: Integer grid

Example:

```
0 10 0 100
0 0 0 0
0 -100 0 0
0 0 0 0
```

### Format B: Symbolic

Example:

```
A . B #
. B . .
. . . L
. . . .
```

Symbols translate to:

* `A` → agent start (cell becomes empty)
* `.` → empty
* `B` → box
* `#` → barrier
* `L` → lava

If map_path is None, the environment generates a **random map** using the parameters:

* number_of_boxes
* number_of_barriers
* number_of_lavas

---

# 10. Step Function Summary

The step(action) method returns:

obs, reward, done, info


Where:

* reward only reflects lava destruction bonuses
* done = agent out of stamina, timestep limit, or no boxes left
* info contains:

  * `"timestep"`
  * `"stamina"`
  * `"number_of_boxes"`
  * `"number_destroyed"`
  * `"last_action_valid"`
  * `"chain_length_k"`
  * `"initial_force_charged"`
  * `"lava_destroyed_this_step"`
  * `"perfect_squares_available"`

---

# 11. Reset Behavior

reset():

* Loads map (or generates random one)
* Resets stamina + timestep
* Clears perfect-square registry
* Clears stationary push state

# 12. LLM interactions

I also used vscode autocomplete sometimes.
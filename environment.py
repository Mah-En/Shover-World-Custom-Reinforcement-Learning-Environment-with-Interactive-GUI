import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces



class ShoverWorldEnv(gym.Env):

    metadata = {
        "render_modes": ["human", None],
        "render_fps": 30,
    }

    # grid
    LAVA_VALUE = -100
    EMPTY_VALUE = 0
    BOX_MIN = 1
    BOX_MAX = 10
    BOX_BASE_VALUE = 10  
    BARRIER_VALUE = 100

    def __init__(
        self,
        render_mode=None,
        n_rows=8, #6
        n_cols=8, #6
        max_timestep=400,
        number_of_boxes=5,
        number_of_barriers=4, #3
        number_of_lavas=3, #2
        initial_stamina=1000.0,
        initial_force=40.0,
        unit_force=10.0,
        perf_sq_initial_age=10,
        map_path=None,
        seed=None,
        r_lava=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_timestep = max_timestep

        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas

        self.initial_stamina = initial_stamina
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.perf_sq_initial_age = perf_sq_initial_age

        self.map_path = map_path

        self.r_lava = self.initial_force if r_lava is None else r_lava

        self._seed = seed
        self.rng = np.random.default_rng(seed)


        self.action_space = self._build_action_space()
        self.observ_space = self._build_observ_space()

       
        self.grid = None              # (n_rows, n_cols)
        self.agent_pos = None         # (row, col)
        self.stamina = float(self.initial_stamina)
        self.timestep = 0

        self.prev_selected_pos = None
        self.prev_action = None

        # **** stationary ****
        self._last_pushed_head = None  # (row, col, dir)

        # **** perfect squares ****
        self._square_registry = []    # dict: n, top_left, age
        self._stationary_state = None 

        self.total_destroyed_boxes = 0

        self._renderer = None  

        self.reset()

    # ******
    # Spaces
    # ******
    def _build_action_space(self):
        """
        action = ( (row, col), type )
        type: 0-6
            1: Move Up
            2: Move Right
            3: Move Down
            4: Move Left
            5: Barrier Maker
            6: Hellify
            0: NOOP
        """
        pos_space = spaces.MultiDiscrete([self.n_rows, self.n_cols])
        action_type_space = spaces.Discrete(7)  # 0..6
        return spaces.Tuple((pos_space, action_type_space))

    def _build_observ_space(self):
        """
            grid: int map
            agent: row, col
            stamina: scalar float
            prev_selected_position: row, col
            prev_action: int
        """
        grid_space = spaces.Box(
            low=-100,
            high=100,
            shape=(self.n_rows, self.n_cols),
            dtype=np.int32,
        )
        agent_space = spaces.Box(
            low=0,
            high=max(self.n_rows, self.n_cols) - 1,
            shape=(2,),
            dtype=np.int32,
        )
        stamina_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(),
            dtype=np.float32,
        )
        prev_pos_space = spaces.Box(
            low=0,
            high=max(self.n_rows, self.n_cols) - 1,
            shape=(2,),
            dtype=np.int32,
        )
        prev_action_space = spaces.Discrete(7) 

        return spaces.Dict(
            {
                "grid": grid_space,
                "agent": agent_space,
                "stamina": stamina_space,
                "prev_selected_position": prev_pos_space,
                "prev_action": prev_action_space,
            }
        )

    # ******
    # Map (A/B)
    # A-> integer 
    # B-> symbolic 
    # ******
    def _load_map_from_file(self, path):

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        first = lines[0].split()
        is_int = True
        for tok in first:
            try:
                int(tok)
            except ValueError:
                is_int = False
                break

        if is_int:
            return self._parse_format_a(lines)
        else:
            return self._parse_format_b(lines)

    def _parse_format_a(self, lines):
        data = []
        for line in lines:
            row = [int(tok) for tok in line.split()]
            data.append(row)

        n_rows = len(data)
        n_cols = len(data[0])

        grid = np.array(data, dtype=np.int32)

        agent_pos = None
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r, c] == self.EMPTY_VALUE:
                    agent_pos = (r, c)
                    break
            if agent_pos is not None:
                break

        return grid, agent_pos

    def _parse_format_b(self, lines):
        """
            . empty -> 0
            B box   -> 10
            # barrier -> 100
            L lava  -> -100
        """
        mapping = {
            ".": self.EMPTY_VALUE,
            "B": self.BOX_BASE_VALUE,
            "#": self.BARRIER_VALUE,
            "L": self.LAVA_VALUE,
            "A": self.EMPTY_VALUE,
        }

        data = []
        agent_pos = None
        n_cols = None

        for r, line in enumerate(lines):
            tokens = line.split() if " " in line else list(line)

            if n_cols is None:
                n_cols = len(tokens)

            row_vals = []
            for c, ch in enumerate(tokens):
                val = mapping[ch]
                row_vals.append(val)
                if ch == "A":
                    agent_pos = (r, c)
            data.append(row_vals)


        grid = np.array(data, dtype=np.int32)
        return grid, agent_pos

    # ******
    # reset
    # ******
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)


        if self.map_path is not None:
            self.grid, self.agent_pos = self._load_map_from_file(self.map_path)
        else:
            self.grid, self.agent_pos = self._gen_random_map()


        self.stamina = float(self.initial_stamina)
        self.timestep = 0
        self.total_destroyed_boxes = 0

        self.prev_selected_pos = self.agent_pos
        self.prev_action = 0  # NOOP

        self._square_registry = []
        self._stationary_state = None
        self._last_pushed_head = None

        obs = self._build_observ()
        info = self._build_info_dict(
            last_action_valid=True,
            chain_length=0,
            initial_force_charged=False,
            lava_destroyed_this_step=0,
        )
        return obs, info


    def _direction_from_action(self, action_type):
        """
        1: Up, 2: Right, 3: Down, 4: Left
        output: dr, dc
        """
        if action_type == 1:
            return -1, 0
        elif action_type == 2:
            return 0, 1
        elif action_type == 3:
            return 1, 0
        elif action_type == 4:
            return 0, -1
        else:
            return 0, 0  

    def _is_within_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _cell_value(self, r, c):
        return int(self.grid[r, c])

    def _is_box(self, val):
        return self.BOX_MIN <= val <= self.BOX_MAX

    # ******
    # Move / Chain Push Logic
    # ******
    def move_or_push(self, dr, dc):

        ar, ac = self.agent_pos
        nr, nc = ar + dr, ac + dc

        if not self._is_within_bounds(nr, nc):
            return False, 0, 0

        next_val = self._cell_value(nr, nc)

        # invalid
        if next_val == self.BARRIER_VALUE or next_val == self.LAVA_VALUE:
            return False, 0, 0

        if next_val == self.EMPTY_VALUE:
            self.agent_pos = (nr, nc)
            return True, 0, 0

        # chain push
        if self._is_box(next_val):
            chain_cells = []
            cr, cc = nr, nc
            while self._is_within_bounds(cr, cc) and self._is_box(self._cell_value(cr, cc)):
                chain_cells.append((cr, cc))
                cr += dr
                cc += dc

            k = len(chain_cells)

            if not self._is_within_bounds(cr, cc):
                return False, k, 0

            after_chain_val = self._cell_value(cr, cc)

            if self._is_box(after_chain_val) or after_chain_val == self.BARRIER_VALUE:
                return False, k, 0

            lava_destroyed = 0

            for (br, bc) in reversed(chain_cells):
                tr, tc = br + dr, bc + dc  

                if self._cell_value(tr, tc) == self.LAVA_VALUE:
                    lava_destroyed += 1
                else:
                    self.grid[tr, tc] = self.grid[br, bc]

                self.grid[br, bc] = self.EMPTY_VALUE

            self.agent_pos = (nr, nc)

            self.total_destroyed_boxes += lava_destroyed

            return True, k, lava_destroyed

        return False, 0, 0

    # ******
    # Perfect square registry + auto-dissolve
    # ******
    def _is_box_cell(self, r, c):
        return self.BOX_MIN <= self.grid[r, c] <= self.BOX_MAX

    def _find_perfect_squares(self):
        squares = []
        max_n = min(self.n_rows, self.n_cols)

        for n in range(2, max_n + 1):
            for r in range(0, self.n_rows - n + 1):
                for c in range(0, self.n_cols - n + 1):
                    sub = self.grid[r:r + n, c:c + n]
                    if not np.all((sub >= self.BOX_MIN) & (sub <= self.BOX_MAX)):
                        continue

                    ok = True
                    for rr in range(r - 1, r + n + 1):
                        for cc in range(c - 1, c + n + 1):
                            if (r <= rr < r + n) and (c <= cc < c + n):
                                # inside
                                continue
                            if not self._is_within_bounds(rr, cc):
                                continue
                            if self._is_box_cell(rr, cc):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue

                    squares.append({"n": n, "top_left": (r, c)})

        return squares

    def _update_registry_and_dissolve(self):

        current = self._find_perfect_squares()
        current_keys = {(sq["n"], sq["top_left"]) for sq in current}

        # age++
        new_registry = []
        for entry in self._square_registry:
            key = (entry["n"], entry["top_left"])
            if key in current_keys:
                entry["age"] += 1
                new_registry.append(entry)

        old_keys = {(e["n"], e["top_left"]) for e in new_registry}
        for sq in current:
            key = (sq["n"], sq["top_left"])
            if key not in old_keys:
                new_registry.append(
                    {"n": sq["n"], "top_left": sq["top_left"], "age": 0}
                )

        self._square_registry = new_registry

        # auto-dissolve
        to_dissolve = [e for e in self._square_registry
                       if e["age"] >= self.perf_sq_initial_age]

        for e in to_dissolve:
            n = e["n"]
            r, c = e["top_left"]
            self.grid[r:r + n, c:c + n] = self.EMPTY_VALUE

        self._square_registry = [
            e for e in self._square_registry if e not in to_dissolve
        ]

    # ******
    # step
    # ******
    def step(self, action):
        """
        action: ((row, col), action_type)
        -> obs, reward, done, info
        """
        (target_pos, action_type) = action
        target_row, target_col = int(target_pos[0]), int(target_pos[1])
        action_type = int(action_type)

        self.prev_selected_pos = (target_row, target_col)
        self.prev_action = action_type

        self.timestep += 1

        # baseline cost  
        self.stamina -= 1.0

        reward = 0.0
        last_action_valid = True
        chain_length = 0
        lava_destroyed_this_step = 0
        initial_force_charged = False

        # ******
        # action 1-4
        # ******
        if action_type in (1, 2, 3, 4):
            dr, dc = self._direction_from_action(action_type)

            if action_type == 1:
                direction_token = "U"
            elif action_type == 2:
                direction_token = "R"
            elif action_type == 3:
                direction_token = "D"
            elif action_type == 4:
                direction_token = "L"

            head_r_before = self.agent_pos[0] + dr
            head_c_before = self.agent_pos[1] + dc

            last_action_valid, chain_length, lava_destroyed_this_step = \
                self.move_or_push(dr, dc)

            if last_action_valid and chain_length > 0:
                head_key = (head_r_before, head_c_before, direction_token)

                stationary = (
                    self._last_pushed_head is None or
                    self._last_pushed_head != head_key
                )

                if stationary:
                    push_cost = self.initial_force + self.unit_force * chain_length
                    initial_force_charged = True
                else:
                    push_cost = self.unit_force * chain_length

                self.stamina -= push_cost
                self._last_pushed_head = head_key

                if lava_destroyed_this_step > 0:
                    refund = self.initial_force * lava_destroyed_this_step
                    self.stamina += refund
                    reward += self.r_lava * lava_destroyed_this_step
            else:
                self._last_pushed_head = None

        # ******
        # Barrier Maker (5)
        # ******
        elif action_type == 5:
            candidates = [e for e in self._square_registry if e["n"] >= 2]
            if not candidates:
                last_action_valid = False
            else:
                # max age
                e = max(candidates, key=lambda x: x["age"])
                n = e["n"]
                r, c = e["top_left"]

                # all
                self.grid[r:r + n, c:c + n] = self.BARRIER_VALUE
                self.stamina += n * n

                self._square_registry = [
                    x for x in self._square_registry if x is not e
                ]

            self._last_pushed_head = None

        # ******
        # Hellify (6)
        # ******
        elif action_type == 6:
            candidates = [e for e in self._square_registry if e["n"] > 2]
            if not candidates:
                last_action_valid = False
            else:
                e = max(candidates, key=lambda x: x["age"])
                n = e["n"]
                r, c = e["top_left"]


                for rr in range(r, r + n):
                    for cc in range(c, c + n):
                        is_border = (
                            rr == r or rr == r + n - 1 or
                            cc == c or cc == c + n - 1
                        )
                        if is_border:
                            self.grid[rr, cc] = self.EMPTY_VALUE
                        else:
                            if self._is_box_cell(rr, cc):
                                self.total_destroyed += 1
                            self.grid[rr, cc] = self.LAVA_VALUE

                self._square_registry = [
                    x for x in self._square_registry if x is not e
                ]

            self._last_pushed_head = None

        else:
            last_action_valid = False
            self._last_pushed_head = None

        self._update_registry_and_dissolve()

        # ******
        # terminate
        # ******
        num_boxes = int(np.sum(
            (self.grid >= self.BOX_MIN) & (self.grid <= self.BOX_MAX)
        ))

        done = False
        if self.stamina <= 0 or self.timestep >= self.max_timestep:
            done = True
        if num_boxes == 0:
            done = True

        obs = self._build_observ()
        info = self._build_info_dict(
            last_action_valid=last_action_valid,
            chain_length=chain_length,
            initial_force_charged=initial_force_charged,
            lava_destroyed_this_step=lava_destroyed_this_step,
        )
        return obs, reward, done, info

    # ******
    # Info
    # ******
    def _build_observ(self):
        grid_obs = self.grid.astype(np.int32)

        agent_row, agent_col = self.agent_pos
        agent_arr = np.array([agent_row, agent_col], dtype=np.int32)

        prev_row, prev_col = self.prev_selected_pos
        prev_pos_arr = np.array([prev_row, prev_col], dtype=np.int32)

        obs = {
            "grid": grid_obs,
            "agent": agent_arr,
            "stamina": np.array(self.stamina, dtype=np.float32),
            "prev_selected_pos": prev_pos_arr,
            "prev_action": int(self.prev_action),
        }
        return obs

    def _build_info_dict(
        self,
        last_action_valid,
        chain_length,
        initial_force_charged,
        lava_destroyed_this_step,
    ):
        num_boxes = int(np.sum(
            (self.grid >= self.BOX_MIN) & (self.grid <= self.BOX_MAX)
        ))
        squares = self._find_perfect_squares()
        info = {
            "timestep": self.timestep,
            "stamina": self.stamina,
            "number_of_boxes": num_boxes,
            "number_destroyed": self.total_destroyed_boxes,
            "last_action_valid": last_action_valid,
            "chain_length_k": chain_length,
            "initial_force_charged": initial_force_charged,
            "lava_destroyed_this_step": lava_destroyed_this_step,
            "perfect_squares_available": [
                (sq["n"], sq["top_left"]) for sq in squares
            ],
        }
        return info

    # ******
    # Random map generation
    # ******
    def _gen_random_map(self):
        grid = np.full(
            (self.n_rows, self.n_cols),
            self.EMPTY_VALUE,
            dtype=np.int32
        )

        all_cells = [(r, c) for r in range(self.n_rows)
                     for c in range(self.n_cols)]
        self.rng.shuffle(all_cells)

        agent_pos = all_cells.pop()

        def place_objects(value, count):
            placed = 0
            while placed < count and all_cells:
                r, c = all_cells.pop()
                if grid[r, c] == self.EMPTY_VALUE:
                    grid[r, c] = value
                    placed += 1

        place_objects(self.BOX_BASE_VALUE, self.number_of_boxes)
        place_objects(self.BARRIER_VALUE, self.number_of_barriers)
        place_objects(self.LAVA_VALUE, self.number_of_lavas)

        return grid, agent_pos
    

    
    def close(self):
        if self._renderer is not None:
            self._renderer = None
        super().close()

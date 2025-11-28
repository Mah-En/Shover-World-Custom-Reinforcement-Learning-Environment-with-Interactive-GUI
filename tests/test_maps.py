import os
from environment import ShoverWorldEnv


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAP_DIR = os.path.join(BASE_DIR, "maps")


def test_load_map_format_a():
    path = os.path.join(MAP_DIR, "simple_map_a.txt")
    env = ShoverWorldEnv(
        map_path=path,
        n_rows=4,
        n_cols=4,
        seed=0,
    )
    obs, info = env.reset()

    assert env.grid.shape == (4, 4)
    ar, ac = env.agent_pos
    assert env.grid[ar, ac] == env.EMPTY_VALUE


def test_load_map_format_b():
    path = os.path.join(MAP_DIR, "simple_map_b.txt")
    env = ShoverWorldEnv(
        map_path=path,
        n_rows=4,
        n_cols=4,
        seed=0,
    )
    obs, info = env.reset()

    assert env.agent_pos == (0, 0)
    assert env.grid[0, 3] == env.BARRIER_VALUE
    assert env.grid[2, 3] == env.LAVA_VALUE

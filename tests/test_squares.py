import numpy as np
from environment import ShoverWorldEnv


def test_find_perfect_square_2x2():
    env = ShoverWorldEnv(
        n_rows=4,
        n_cols=4,
        initial_stamina=100.0,
        initial_force=5.0,
        unit_force=1.0,
        perf_sq_initial_age=3,
        max_timestep=50,
        map_path=None,
        seed=0,
    )

    env.grid = np.array(
        [
            [10, 10, 0, 0],
            [10, 10, 0, 0],
            [0,  0,  0, 0],
            [0,  0,  0, 0],
        ],
        dtype=np.int32,
    )
    env.agent_pos = (3, 3)

    squares = env._find_perfect_squares()
    assert len(squares) == 1
    assert squares[0]["n"] == 2
    assert squares[0]["top_left"] == (0, 0)


def test_auto_dissolve_after_age():
    env = ShoverWorldEnv(
        n_rows=3,
        n_cols=3,
        initial_stamina=100.0,
        initial_force=5.0,
        unit_force=1.0,
        perf_sq_initial_age=2,  
        max_timestep=50,
        map_path=None,
        seed=0,
    )

    env.grid = np.array(
        [
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10],
        ],
        dtype=np.int32,
    )
    env.agent_pos = (1, 1)
    env._square_registry = []

    for _ in range(3):
        env._update_square_registry_and_dissolve()

    assert (env.grid == env.EMPTY_VALUE).all()

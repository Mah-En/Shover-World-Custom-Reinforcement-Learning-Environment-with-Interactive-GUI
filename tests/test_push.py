import numpy as np
from environment import ShoverWorldEnv


def make_env_for_push():
    env = ShoverWorldEnv(
        n_rows=4,
        n_cols=4,
        initial_stamina=100.0,
        initial_force=5.0,
        unit_force=1.0,
        perf_sq_initial_age=10,
        max_timestep=50,
        map_path=None,
        seed=0,
    )

    env.grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 10, 10, 0], 
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    env.agent_pos = (1, 0)
    env.stamina = 100.0
    env.timestep = 0
    env.previous_selected_position = env.agent_pos
    env.previous_action = 0
    env._square_registry = []
    env._last_pushed_head = None
    return env


def test_simple_chain_push_valid():
    env = make_env_for_push()

    action = ((env.agent_pos[0], env.agent_pos[1]), 2)  # 2 = Right
    obs, reward, done, info = env.step(action)

    row1 = env.grid[1, :]
    assert (row1 == np.array([0, 0, 10, 10])).all()

    assert env.agent_pos == (1, 1)

    assert info["chain_length_k"] == 2
    assert info["last_action_valid"] is True


def test_push_blocked_by_barrier():
    env = make_env_for_push()
    env.grid[1, 3] = env.BARRIER_VALUE

    action = ((env.agent_pos[0], env.agent_pos[1]), 2)  # Right
    obs, reward, done, info = env.step(action)

    row1 = env.grid[1, :]
    assert (row1 == np.array([0, 10, 10, env.BARRIER_VALUE])).all()
    assert info["last_action_valid"] is False
    assert info["chain_length_k"] == 2 

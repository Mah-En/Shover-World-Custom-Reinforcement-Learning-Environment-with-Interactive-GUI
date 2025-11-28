import numpy as np
from environment import ShoverWorldEnv


def make_env_for_stamina():
    env = ShoverWorldEnv(
        n_rows=3,
        n_cols=4,
        initial_stamina=100.0,
        initial_force=10.0,
        unit_force=2.0,
        perf_sq_initial_age=10,
        max_timestep=50,
        map_path=None,
        seed=0,
    )

    # grid:
    # A = agent, B = box, L = lava
    # [A, B, B, L]
    env.grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 10, 10, -100],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    env.agent_pos = (1, 0)
    env.stamina = 100.0
    env.timestep = 0
    env.previous_selected_position = env.agent_pos
    env.previous_action = 0
    env._last_pushed_head = None
    env._square_registry = []
    return env


def test_baseline_cost_only():
    env = make_env_for_stamina()
    action = ((env.agent_pos[0], env.agent_pos[1]), 1)  # Up
    old_stamina = env.stamina
    obs, reward, done, info = env.step(action)

    assert env.stamina == old_stamina - 1.0


def test_push_and_lava_refund():
    env = make_env_for_stamina()

    old_stamina = env.stamina
    action = ((env.agent_pos[0], env.agent_pos[1]), 2)  # Right
    obs, reward, done, info = env.step(action)

    expected = old_stamina - 1 - 14 + 10
    assert env.stamina == expected

    assert info["chain_length_k"] == 2
    assert info["lava_destroyed_this_step"] == 1
    assert info["initial_force_charged"] is True

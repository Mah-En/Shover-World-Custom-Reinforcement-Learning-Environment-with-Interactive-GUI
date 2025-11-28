import numpy as np
from environment import ShoverWorldEnv


def main():
    env = ShoverWorldEnv(
        n_rows=6,
        n_cols=6,
        initial_stamina=200.0,
        initial_force=10.0,
        unit_force=2.0,
        perf_sq_initial_age=5,
        max_timestep=200,
        map_path=None,  # random
        seed=0,
    )

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

    print("Episode finished.")
    print("  Steps:", step_count)
    print("  Total reward:", total_reward)
    print("  Boxes destroyed:", info.get("number_destroyed", None))

    env.close()


if __name__ == "__main__":
    main()

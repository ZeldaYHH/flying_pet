import gymnasium as gym

import flying_pet  # noqa: F401 - registers FlyingPet-v0


def main():
    env = gym.make("FlyingPet-v0")
    observation, _ = env.reset(seed=0)
    print("initial_observation:", observation)

    for step in range(200):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        print(
            f"step={step + 1:03d}",
            f"reward={reward:.3f}",
            "observation=",
            observation,
        )

        if terminated or truncated:
            observation, _ = env.reset()
            print("reset_observation:", observation)

    env.close()


if __name__ == "__main__":
    main()

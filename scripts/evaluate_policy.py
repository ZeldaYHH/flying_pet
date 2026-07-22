from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import flying_pet  # noqa: F401 - registers FlyingPet-v0


def main():
    model_path = Path("runs/ppo_hover/flying_pet_ppo.zip")

    render_mode = "human" if "--render" in __import__("sys").argv else None
    env = gym.make("FlyingPet-v0", render_mode=render_mode)
    model = PPO.load(model_path)

    observation, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    print(f"total_reward: {total_reward:.3f}")
    print("final_observation:", observation)

    env.close()


if __name__ == "__main__":
    main()

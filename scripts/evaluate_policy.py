from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import flying_pet  # noqa: F401 - registers FlyingPet-v0


def main():
    model_path = Path("runs/ppo_hover/flying_pet_ppo.zip")

    render_mode = "human" if "--render" in __import__("sys").argv else None
    env_kwargs = {}
    if render_mode == "human":
        # Ten minutes at 100 control steps per second. During visualization,
        # only an R keypress or this time limit resets the vehicle.
        env_kwargs = {
            "episode_step_limit": 60_000,
            "max_xy_distance": None,
            "min_height": None,
            "max_height": None,
        }
    env = gym.make("FlyingPet-v0", render_mode=render_mode, **env_kwargs)
    model = PPO.load(model_path)

    episode = 0
    while True:
        observation, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            viewer = env.unwrapped.viewer
            if viewer is not None and not viewer.is_running():
                env.close()
                return

        episode += 1
        print(f"episode={episode} total_reward={total_reward:.3f}")
        print("final_observation:", observation)

        # Rendering loops episodes until the MuJoCo window is closed.
        if render_mode != "human":
            break

    env.close()


if __name__ == "__main__":
    main()

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import flying_pet  # noqa: F401 - registers FlyingPet-v0


def main():
    run_dir = Path("runs/ppo_hover")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("FlyingPet-v0")

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=str(run_dir),
        verbose=1,
    )
    model.learn(total_timesteps=200_000)
    model.save(run_dir / "flying_pet_ppo.zip")

    env.close()


if __name__ == "__main__":
    main()

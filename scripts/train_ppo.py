import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import flying_pet  # noqa: F401 - registers FlyingPet-v0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Number of environment steps to train (default: 200000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from runs/ppo_hover/flying_pet_ppo.zip.",
    )
    args = parser.parse_args()

    run_dir = Path("runs/ppo_hover")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("FlyingPet-v0")

    model_path = run_dir / "flying_pet_ppo.zip"
    if args.resume:
        model = PPO.load(model_path, env=env, tensorboard_log=str(run_dir), verbose=1)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=str(run_dir),
            verbose=1,
        )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)
    model.save(model_path)

    env.close()


if __name__ == "__main__":
    main()

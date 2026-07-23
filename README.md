# FlyingPet

FlyingPet is a minimal quadrotor-like reinforcement learning environment built
with [Gymnasium](https://gymnasium.farama.org/) and
[MuJoCo](https://mujoco.org/). It models a free rigid body controlled by total
thrust and body torques, with a simple hover task at `z = 1.0`.

The repository also includes a small
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO training
example.

## Features

- Gymnasium environment registered as `FlyingPet-v0`
- MuJoCo free-body dynamics with a lightweight quadrotor model
- Continuous thrust and torque control
- Hover reward based on position, velocity, angular velocity, and attitude
- PPO training and evaluation scripts
- Human and RGB-array rendering modes

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ZeldaYHH/flying_pet.git
cd flying_pet

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To inspect training progress with TensorBoard, install it separately:

```bash
python -m pip install tensorboard
```

## Quick Start

Importing `flying_pet` registers the environment with Gymnasium:

```python
import gymnasium as gym
import flying_pet

env = gym.make("FlyingPet-v0")
observation, info = env.reset(seed=0)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Run the random-action smoke test:

```bash
python -m scripts.test_env
```

The script runs 200 steps and prints each observation and reward.

## Environment

### Observation

The observation is a 13-element `float32` vector:

| Indices | Quantity | Size |
| --- | --- | ---: |
| `0:3` | Position `[x, y, z]` | 3 |
| `3:6` | Linear velocity `[vx, vy, vz]` | 3 |
| `6:10` | Quaternion `[w, x, y, z]` | 4 |
| `10:13` | Angular velocity `[wx, wy, wz]` | 3 |

### Action

The action is a 4-element `float32` vector:

| Index | Control | Range |
| ---: | --- | ---: |
| `0` | Total thrust | `[0, 20]` |
| `1` | Roll torque | `[-2, 2]` |
| `2` | Pitch torque | `[-2, 2]` |
| `3` | Yaw torque | `[-1, 1]` |

Actions are clipped to these limits before being applied in the body frame.

### Reward

The reward encourages the body to hover near the origin at `z = 1.0`, remain
upright, and move slowly:

```text
reward = 1
         - 2.00 * (z - 1)^2
         - 0.50 * ||position_xy||^2
         - 0.10 * ||linear_velocity||^2
         - 0.05 * ||angular_velocity||^2
         + 0.50 * upright
```

An episode terminates when the body falls below the floor, rises above
`z = 5`, or moves more than 5 units from the origin in the horizontal plane.
It is truncated after 500 steps.

## Train PPO

The training script uses Stable-Baselines3 PPO with `MlpPolicy`. The default
model is intentionally small:

| Component | Architecture |
| --- | --- |
| Policy network | `13 -> 64 -> 64 -> 4` |
| Value network | `13 -> 64 -> 64 -> 1` |
| Hidden activation | `Tanh` |

The policy and value networks use separate hidden layers. Start training with:

```bash
python -m scripts.train_ppo
```

It trains for 200,000 timesteps and saves the model to:

```text
runs/ppo_hover/flying_pet_ppo.zip
```

Training outputs are ignored by Git and remain local.

Launch TensorBoard in another terminal:

```bash
tensorboard --logdir runs/ppo_hover
```

## Evaluate

Evaluate the saved model for one deterministic episode:

```bash
python -m scripts.evaluate_policy
```

The script prints the total reward and final observation. Enable optional
MuJoCo rendering with:

```bash
python -m scripts.evaluate_policy --render
```

## Project Structure

```text
.
|-- flying_pet/
|   |-- __init__.py
|   `-- envs/
|       |-- __init__.py
|       `-- flying_pet_env.py
|-- scripts/
|   |-- evaluate_policy.py
|   |-- test_env.py
|   `-- train_ppo.py
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## License

This project is available under the [MIT License](LICENSE).

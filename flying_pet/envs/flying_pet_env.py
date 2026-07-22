from __future__ import annotations

import importlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


MODEL_XML = """
<mujoco model="flying_pet">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <geom rgba="0.35 0.55 0.95 1" contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"
          contype="1" conaffinity="1"/>

    <body name="pet" pos="0 0 1">
      <freejoint/>
      <geom name="body" type="box" size="0.18 0.12 0.04" mass="1.0"
            rgba="0.2 0.45 0.95 1"/>
      <geom name="arm_x" type="capsule" fromto="-0.35 0 0 0.35 0 0" size="0.015"
            mass="0.04" rgba="0.1 0.1 0.1 1"/>
      <geom name="arm_y" type="capsule" fromto="0 -0.35 0 0 0.35 0" size="0.015"
            mass="0.04" rgba="0.1 0.1 0.1 1"/>
      <geom name="rotor_front" type="cylinder" pos="0 0.35 0" size="0.08 0.01"
            mass="0.02" rgba="0.1 0.8 0.5 1"/>
      <geom name="rotor_back" type="cylinder" pos="0 -0.35 0" size="0.08 0.01"
            mass="0.02" rgba="0.1 0.8 0.5 1"/>
      <geom name="rotor_left" type="cylinder" pos="-0.35 0 0" size="0.08 0.01"
            mass="0.02" rgba="0.9 0.45 0.2 1"/>
      <geom name="rotor_right" type="cylinder" pos="0.35 0 0" size="0.08 0.01"
            mass="0.02" rgba="0.9 0.45 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class FlyingPetEnv(gym.Env):
    """A minimal MuJoCo quadrotor-like free rigid body environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, frame_skip: int = 5):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = 500
        self.step_count = 0

        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pet")

        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0, -2.0, -1.0], dtype=np.float32),
            high=np.array([20.0, 2.0, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_limit = np.inf * np.ones(13, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

        self.viewer = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.data.qpos[:3] = np.array([0.0, 0.0, 1.0])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0

        if options and options.get("randomize", False):
            self.data.qpos[:3] += self.np_random.uniform(-0.05, 0.05, size=3)
            self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=6)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        total_thrust, roll_torque, pitch_torque, yaw_torque = action

        self.data.xfrc_applied[:] = 0.0
        force_world = self._body_vector_to_world(np.array([0.0, 0.0, total_thrust]))
        torque_world = self._body_vector_to_world(
            np.array([roll_torque, pitch_torque, yaw_torque])
        )
        self.data.xfrc_applied[self.body_id, :3] = force_world
        self.data.xfrc_applied[self.body_id, 3:] = torque_world

        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward(obs)

        position = obs[:3]
        terminated = bool(
            position[2] < 0.0 or position[2] > 5.0 or np.linalg.norm(position[:2]) > 5.0
        )
        truncated = self.step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model)
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data)
            return renderer.render()

        if self.render_mode == "human":
            try:
                viewer = importlib.import_module("mujoco.viewer")
            except ImportError as exc:
                raise RuntimeError("MuJoCo viewer is not available.") from exc

            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        position = self.data.qpos[:3]
        quaternion = self.data.qpos[3:7]
        linear_velocity = self.data.qvel[:3]
        angular_velocity = self.data.qvel[3:6]
        return np.concatenate(
            [position, linear_velocity, quaternion, angular_velocity]
        ).astype(np.float32)

    def _reward(self, obs: np.ndarray) -> float:
        position = obs[:3]
        linear_velocity = obs[3:6]
        angular_velocity = obs[10:13]
        upright = self._body_z_axis_world()[2]

        z_error = position[2] - 1.0
        xy_error = np.linalg.norm(position[:2])

        reward = 1.0
        reward -= 2.0 * z_error**2
        reward -= 0.5 * xy_error**2
        reward -= 0.1 * np.dot(linear_velocity, linear_velocity)
        reward -= 0.05 * np.dot(angular_velocity, angular_velocity)
        reward += 0.5 * upright
        return float(reward)

    def _body_vector_to_world(self, vector: np.ndarray) -> np.ndarray:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.data.qpos[3:7])
        return matrix.reshape(3, 3) @ vector

    def _body_z_axis_world(self) -> np.ndarray:
        return self._body_vector_to_world(np.array([0.0, 0.0, 1.0]))

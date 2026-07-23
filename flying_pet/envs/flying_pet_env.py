from __future__ import annotations

import importlib
import time

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
    <geom name="floor" type="plane" size="50 50 0.1" rgba="0.9 0.9 0.9 1"
          contype="1" conaffinity="1"/>

    <!-- World-frame reference: X red, Y green, Z blue. -->
    <geom name="origin" type="sphere" pos="0 0 0.025" size="0.055"
          rgba="1 1 1 1" contype="0" conaffinity="0"/>
    <geom name="axis_x" type="capsule" fromto="0 0 0.03 2 0 0.03" size="0.018"
          rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>
    <geom name="axis_y" type="capsule" fromto="0 0 0.03 0 2 0.03" size="0.018"
          rgba="0.1 0.75 0.15 1" contype="0" conaffinity="0"/>
    <geom name="axis_z" type="capsule" fromto="0 0 0.03 0 0 2" size="0.018"
          rgba="0.1 0.3 0.95 1" contype="0" conaffinity="0"/>

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

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        episode_step_limit: int = 500,
        max_xy_distance: float | None = 5.0,
        min_height: float | None = 0.0,
        max_height: float | None = 5.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.episode_step_limit = episode_step_limit
        self.max_xy_distance = max_xy_distance
        self.min_height = min_height
        self.max_height = max_height
        self.step_count = 0
        self._reset_requested = False

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
        self.renderer = None
        self._last_render_time = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._reset_requested = False

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

        # In passive-viewer mode, sync() writes the current mouse perturbation
        # into xfrc_applied after the preceding physics step. Preserve it before
        # replacing the policy force so both affect the next mj_step().
        mouse_perturbation = np.zeros(6)
        if self.render_mode == "human" and self.viewer is not None:
            mouse_perturbation = self.data.xfrc_applied[self.body_id].copy()

        self.data.xfrc_applied[:] = 0.0
        force_world = self._body_vector_to_world(np.array([0.0, 0.0, total_thrust]))
        torque_world = self._body_vector_to_world(
            np.array([roll_torque, pitch_torque, yaw_torque])
        )
        self.data.xfrc_applied[self.body_id, :3] = (
            force_world + mouse_perturbation[:3]
        )
        self.data.xfrc_applied[self.body_id, 3:] = (
            torque_world + mouse_perturbation[3:]
        )

        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward(obs)

        position = obs[:3]
        terminated = False
        if self.min_height is not None:
            terminated = bool(position[2] < self.min_height)
        if self.max_height is not None:
            terminated = terminated or bool(position[2] > self.max_height)
        if self.max_xy_distance is not None:
            terminated = terminated or bool(
                np.linalg.norm(position[:2]) > self.max_xy_distance
            )
        truncated = (
            self.step_count >= self.episode_step_limit or self._reset_requested
        )

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

        if self.render_mode == "human":
            try:
                viewer = importlib.import_module("mujoco.viewer")
            except ImportError as exc:
                raise RuntimeError("MuJoCo viewer is not available.") from exc

            if self.viewer is None:
                self.viewer = viewer.launch_passive(
                    self.model, self.data, key_callback=self._key_callback
                )
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = self.body_id
                self.viewer.cam.distance = 6.0
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -25.0
                self._last_render_time = time.monotonic()
            # Remove the just-applied policy force before sync() writes the
            # viewer's mouse perturbation for the following physics step.
            self.data.xfrc_applied[:] = 0.0
            self.viewer.sync()
            # Keep playback close to real time instead of racing through an episode.
            target_period = self.model.opt.timestep * self.frame_skip
            now = time.monotonic()
            if self._last_render_time is not None:
                time.sleep(max(0.0, target_period - (now - self._last_render_time)))
            self._last_render_time = time.monotonic()
            return None

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

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

    def _key_callback(self, keycode: int):
        if keycode in (ord("R"), ord("r")):
            self._reset_requested = True

import gymnasium as gym
import gymnasium_robotics

from sb3_contrib import TQC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPushDense-v4"

env = gym.make(ENV_ID, max_episode_steps=150)
env = Monitor(env)

model = TQC(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=2048,
    gamma=0.95,
    tau=0.05,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10_000,
    verbose=1,
    policy_kwargs=dict(
        net_arch=[512, 512, 512],
        n_critics=2,
    ),
    tensorboard_log="./tensorboard/",
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="fetchpush_tqc"
)

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback,
    progress_bar=True,
)

model.save("fetchpush_v4_tqc_her")
env.close()

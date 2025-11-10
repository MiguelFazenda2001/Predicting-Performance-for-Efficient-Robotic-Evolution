import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3",continuous=True,enable_wind=True,wind_power=15.0,turbulence_power=1.5, render_mode="human")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(500):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
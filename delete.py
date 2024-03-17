import logging

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

training_period = 20  # record the agent's episode every 250
num_training_episodes = 40  # total number of training episodes

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

for episode_num in range(num_training_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # replace with actual agent
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    logging.info(f"episode-{episode_num}", info["episode"])
env.close()
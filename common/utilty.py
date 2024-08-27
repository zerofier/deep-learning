import gymnasium as gym


def record_video(env, agent, video_folder):
    wrapper = gym.wrappers.RecordVideo(env, video_folder)
    observation, info = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        sample = agent.action_sample(observation)
        action = sample[0][0]
        observation, reward, terminated, truncated, info = wrapper.step(action)
        wrapper.render()

    wrapper.close()

import gymnasium as gym
import torch
import numpy as np
from DQNagent import DQNAgent


def evaluate(env_name='LunarLander-v3',weights_path='dqn_lunarlander.pth',
             n_episodes=100,render=False):
    render_mode='human' if render else None
    env=gym.make(env_name,render_mode=render_mode)
    obs_dim=env.observation_space.shape[0]
    n_actions=env.action_space.n

    agent=DQNAgent(obs_dim, n_actions)
    agent.q_online.load_state_dict(torch.load(weights_path, map_location=agent.device))
    agent.q_online.eval()

    rewards=[]
    for episode in range(n_episodes):
        state,_=env.reset()
        episode_reward=0
        done=False

        while not done:
            action=agent.select_action(state, greedy=True)   # no exploration
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            state=next_state
            episode_reward+=reward

        rewards.append(episode_reward)
        print(f"Episode {episode},reward {episode_reward:.1f}")

    rewards=np.array(rewards)
    print(f"\nMean reward over {n_episodes} episodes: {rewards.mean():.1f} "
          f"± {rewards.std():.1f}")
    print(f"Min: {rewards.min():.1f}, Max: {rewards.max():.1f}")
    env.close()
    return rewards


if __name__ == '__main__':
    evaluate(n_episodes=5,render=True)
import gymnasium as gym
import torch
from DQNagent import DQNAgent
from collections import deque
import numpy as np

def train():
    env=gym.make('LunarLander-v3')
    obs_dim=env.observation_space.shape[0]
    n_actions=env.action_space.n
    agent=DQNAgent(obs_dim,n_actions)

    max_episodes=2000
    solved_threshold=200
    batch_size=agent.batch_size
    target_sync_freq=1000
    reward_window=deque(maxlen=100)

    for episode in range(max_episodes):
        state,_=env.reset()
        episode_reward=0
        done=False

        while not done:
            action=agent.select_action(state)
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            agent.store(state,action,reward,next_state,done)

            if len(agent.replay_buffer)>=batch_size:
                agent.update()

            if agent.step_count%target_sync_freq==0:
                agent.sync_target()

            state=next_state
            episode_reward+=reward

        reward_window.append(episode_reward)
        avg_reward=np.mean(reward_window)
        print(f"Episode {episode}, reward {episode_reward:.1f}, avg {avg_reward:.1f}")

        if len(reward_window)==100 and avg_reward>=solved_threshold:
            print(f"Solved at episode {episode},avg{avg_reward:.1f}")
            break

    torch.save(agent.q_online.state_dict(),'dqn_lunarlander.pth')


if __name__=="__main__":
    train()
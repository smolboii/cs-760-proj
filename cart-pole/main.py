import gym
import numpy as np
import torch
from transition import Transition
from agent import Agent
from collections import deque

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, '\n')

    env = gym.make('CartPole-v1')
    agent = Agent(len(env.observation_space.low), env.action_space.n, device)

    agent.model.load_state_dict(torch.load('models\episode=200 avg_reward=495.7.pt'))

    transitions = deque(maxlen=1_000_000)
    num_episodes = 2_000
    chkpt_every = 20
    sum_rewards = []

    batch_size=64
    min_transitions = 1_000

    for ep in range(1,num_episodes):

        state = env.reset()
        done = False
        ep_rewards = []

        while not done:
            action = agent.get_action(state, eval=True)

            new_state, reward, done, _ = env.step(action)
            env.render()
            ep_rewards.append(reward)

            transitions.append(Transition(state, action, new_state, reward, done))
            state = new_state

            # if len(transitions) > min_transitions:
            #     agent.train(np.random.choice(transitions, batch_size, replace=False))

        sum_rewards.append(np.sum(ep_rewards))

        if (ep % chkpt_every == 0):
            avg_reward = np.average(sum_rewards)
            print(f'avg reward: {avg_reward}')
            torch.save(agent.model.state_dict(), f'models/episode={ep} avg_reward={avg_reward}.pt')
            sum_rewards = []

    env.close()
import gym
import numpy as np
import torch
from transition import Transition
from agent import Agent
from collections import deque
import argparse
import os
import time
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-vis', action='store_true', help='runs environment in visualisation mode')
    parser.add_argument('-ex', '--experiment-name', default='def_exp', help='name of the experiment')
    parser.add_argument('-ch', '--checkpoint-path', help='path to the model checkpoint to load')
    parser.add_argument('-n', '--num-episodes', default=2000, type=int, help='number of episodes to run')
    parser.add_argument('-ci', '--checkpoint-interval', default=20, type=int, help='number of episodes between each checkpoint')

    args = parser.parse_args()

    cwd = os.path.dirname(__file__)
    exp_name = args.experiment_name
    exp_dir = os.path.join(cwd, 'experiments', exp_name)
    if args.vis:
        pass
    elif os.path.exists(exp_dir):
        raise Exception(f'Folder with specified experiment name "{exp_name}" already exists for this environment!')
    else:
        os.makedirs(exp_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, '\n')

    env = gym.make('CartPole-v1')
    agent = Agent(len(env.observation_space.low), env.action_space.n, device)

    if args.checkpoint_path != None:
        agent.model.load_state_dict(torch.load(args.checkpoint_path))

    start_time = round(time.time())

    transitions = deque(maxlen=1_000_000)
    num_episodes = args.num_episodes
    chkpt_every = args.checkpoint_interval
    sum_rewards = []

    batch_size=64
    min_transitions = 1_000

    for ep in range(1, num_episodes):

        state = env.reset()
        done = False
        ep_rewards = []

        while not done:
            action = agent.get_action(state, eval=args.vis)

            new_state, reward, done, _ = env.step(action)
            if args.vis:            
                env.render()

            ep_rewards.append(reward)

            transitions.append(Transition(state, action, new_state, reward, done))
            state = new_state

            if len(transitions) > min_transitions and not args.vis:
                agent.train(np.random.choice(transitions, batch_size, replace=False))

        sum_rewards.append(np.sum(ep_rewards))

        if ep % chkpt_every == 0 and not args.vis:

            # save model checkpoint and log training progress

            avg_reward = np.average(sum_rewards)
            time_elapsed = round(time.time()) - start_time
            print(f'ep_num: {ep}, avg reward: {avg_reward}, time_elapsed: {datetime.timedelta(seconds=time_elapsed)}')
            torch.save(agent.model.state_dict(), os.path.join(exp_dir, f'episode={ep} avg_reward={avg_reward}.pt'))

            with open(os.path.join(exp_dir, 'data.txt'), 'a') as data_file:
                data_file.write(f'{ep} {avg_reward} {time_elapsed} \n')

            sum_rewards = []

    env.close()
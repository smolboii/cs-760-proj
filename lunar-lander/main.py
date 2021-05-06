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
import json

def complete_config(config_obj):

    default_config = {
        'inter_neurons': 20,
        'command_neurons': 10,
        'sensory_fanout': 9,
        'inter_fanout': 6,
        'recurrent_command_synapses': 0,
        'motor_fanin': 6,
        'discount_factor': 0.99,
        'epsilon_decay': 0.99975,
        'min_epsilon': 0.01,
        "lr": 0.01,
        "lr_decay": 0.1,
        "lr_decay_interval": 200,
        'batch_size': 64
    }

    # add defaults to config obj if values were not specified
    new_config = default_config.copy()
    new_config.update(config_obj)

    return new_config

if __name__ == '__main__':

    cwd = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument('-vis', action='store_true', help='runs environment in visualisation mode')
    parser.add_argument('-ex', '--experiment-name', default='def_exp', help='name of the experiment')
    parser.add_argument('-cf', '--config-path', default=os.path.join(cwd, 'config.json'), help='path to the model config json file')
    parser.add_argument('-ch', '--checkpoint-path', help='path to the model checkpoint to load')
    parser.add_argument('-n', '--num-episodes', default=600, type=int, help='number of episodes to run')
    parser.add_argument('-ci', '--checkpoint-interval', default=20, type=int, help='number of episodes between each checkpoint')

    args = parser.parse_args()

    with open(args.config_path) as cf:
        config_obj = json.load(cf)

    config_obj = complete_config(config_obj)
    print('\nCompleted configuration object:')
    print(config_obj)

    exp_name = args.experiment_name
    exp_dir = os.path.join(cwd, 'experiments', exp_name)
    if args.vis:
        pass
    elif os.path.exists(exp_dir):
        raise Exception(f'Folder with specified experiment name "{exp_name}" already exists for this environment!')
    else:
        os.makedirs(exp_dir)
        # save config file to experiment folder to describe experiment
        with open(os.path.join(exp_dir, 'config.json'), 'w') as exp_cf:
            json.dump(config_obj, exp_cf, indent=4)  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, '\n')

    env = gym.make('LunarLander-v2')
    agent = Agent(len(env.observation_space.low), env.action_space.n, device, config_obj)

    if args.checkpoint_path != None:
        agent.model.load_state_dict(torch.load(args.checkpoint_path))
        agent.target_model.load_state_dict(agent.model.state_dict())

    start_time = round(time.time())

    transitions = deque(maxlen=1_000_000)
    num_episodes = args.num_episodes
    chkpt_every = args.checkpoint_interval
    sum_rewards = []

    batch_size=config_obj['batch_size']
    min_transitions = 1_000

    iterations = 0
    lr = config_obj['lr']
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
                iterations += 1
                agent.train(np.random.choice(transitions, batch_size, replace=False))

        sum_rewards.append(np.sum(ep_rewards))

        if ep % config_obj['lr_decay_interval'] == 0:
            lr *= config_obj['lr_decay_rate']
            agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

        if ep % chkpt_every == 0 and not args.vis:

            # save model checkpoint and log training progress

            # evaluate model performance (no random actions)
            eval_rewards = []
            for _ in range(chkpt_every):
                state = env.reset()
                done = False

                ep_rewards = []
                while not done:
                    action = agent.get_action(state, eval=True)
                    new_state, reward, done, _ = env.step(action)
                    ep_rewards.append(reward)
                    state = new_state
                eval_rewards.append(np.sum(ep_rewards))
            
            avg_train_reward = np.average(sum_rewards)
            avg_eval_reward = np.average(eval_rewards)

            time_elapsed = round(time.time()) - start_time
            print(f'ep_num: {ep}, avg train reward: {avg_train_reward}, avg eval reward: {avg_eval_reward} time_elapsed: {datetime.timedelta(seconds=time_elapsed)}')
            torch.save(agent.model.state_dict(), os.path.join(exp_dir, f'episode={ep} avg_eval_reward={avg_eval_reward}.pt'))

            with open(os.path.join(exp_dir, 'data.txt'), 'a') as data_file:
                data_file.write(f'{ep} {avg_train_reward} {avg_eval_reward} {time_elapsed} {iterations}\n')

            sum_rewards = []

    env.close()
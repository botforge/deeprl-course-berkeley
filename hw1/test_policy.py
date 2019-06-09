import os
import numpy as np
import gym
import pdb
from models import BaselineNN
import torch

def idx_to_envname(idx):
    dir_path = os.getcwd()
    expert_data_path = dir_path + '/expert_data'
    filenames = os.listdir(expert_data_path)
    env_name = filenames[idx].split('.')[0]
    return env_name

def initialize_net(env, env_name, algo):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    net = BaselineNN(input_dim, output_dim)
    net.load_state_dict(torch.load('/home/dhruvkar/Desktop/Robotics/deeprl/homework_pytorch/hw1/model_dict/' + str(algo) + '-' + env_name))
    net = net.float()
    net.eval()
    return net

def main():
    #using args is very verbose so you can just change the parameters in here
    expert = 2
    algo = 'b'
    num_rollouts = 10
    
    env = gym.make(idx_to_envname(expert))
    net = initialize_net(env, idx_to_envname(expert), algo)
    max_steps = env.spec.timestep_limit
    with torch.no_grad():
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = net(torch.from_numpy(obs[None, :]))
                action = np.expand_dims(action, axis=1)
                obs, r, _, _= env.step(action)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
                env.render()

if __name__ == '__main__':
    main()

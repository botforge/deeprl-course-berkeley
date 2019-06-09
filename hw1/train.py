import os, sys, gym
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
import pickle
import os 
dir_path = os.getcwd()
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DataLoader, TensorDataset
from models import BaselineNN
from tensorboardX import SummaryWriter
import tf_util
import load_policy
import tensorflow as tf
import pdb
#Required for Tensorboard logging stuff
writer = None 

#using args is very verbose, so change parameters up here
config = {'expert':1, 'train_num_rollouts': 3, 'test_num_rollouts': 3,'run_rollouts':True, 'algo':'d', 'epochs':2, 'render':False, 'run_train':True}


def initialize_net(env, env_name, algo):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    net = BaselineNN(input_dim, output_dim)
    net.load_state_dict(torch.load(dir_path + '/model_dict/' + str(config['algo']) +'-'+ env_name))
    net = net.float()
    net.eval()
    return net

def get_expert_policy_fromobs(observations):
    actions = []
    expert_policy_file = 'experts/' + idx_to_envname(config['expert']) + '.pkl'
    policy_fn = load_policy.load_policy(expert_policy_file)
    for i in range(len(observations)):
        obs = observations[i]
        action = policy_fn(obs[None, :])
        actions.append(action)
    # pdb.set_trace()
    return np.array(actions)

def run_best_policy():
    expert_name = idx_to_envname(config['expert'])
    observations = []
    actions = []
    returns = []
    env = gym.make(expert_name)
    net = initialize_net(env, expert_name, config['algo'])
    max_steps = env.spec.timestep_limit
    with torch.no_grad():
        for i in range(config['test_num_rollouts']):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = net(torch.from_numpy(obs[None, :]))
                action = np.expand_dims(action, axis=1)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
                if config['render']:
                    env.render()
            returns.append(totalr)
    #return total reward @ each rollout
    return returns, np.array(observations), np.array(actions)

def idx_to_envname(idx):
    expert_data_path = dir_path + '/expert_data' 
    filenames = os.listdir(expert_data_path) 
    env_name = filenames[idx].split('.')[0] 
    return env_name 

def get_expert_policy(config):
    #get configs from dictionary
    expert = config['expert']
    num_rollouts = config['train_num_rollouts']
    expert_data_path = dir_path + '/expert_data'
    filenames = os.listdir(expert_data_path)
    exp_policy = (filenames[expert]).split('.')[0]
    #run_rollouts
    if(config['run_rollouts']):
        os.system('python run_expert.py experts/' + exp_policy + '.pkl ' + exp_policy +' --num_rollouts ' + str(num_rollouts))

    fn = expert_data_path + '/' + exp_policy + '.pkl'
    with open(fn, 'rb') as f:
        data = pickle.loads(f.read())
    return data['observations'], data['actions']

def log_loss(descriptior, loss_item, epoch):
    global writer
    writer.add_scalar(descriptior, loss_item, epoch)

def loss_pass(net, loss_fn, dataloader, epoch, optimizer=None):
    loss_hist = []
    for i, batch in enumerate(dataloader, 0):
        obs, exp_act = batch
        pred_act = net(obs)
        loss = loss_fn(pred_act, exp_act)

        if optimizer:
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_item = loss.item()

        #saving batch losses
        loss_hist.append(loss_item)

    return loss_hist

def train(net, RLDataset):
    num_epochs = config['epochs']
    num_workers = 4
    batch_size = 8
    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    RLDataloader = DataLoader(RLDataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    best_reward = 0.0

    #save dummy base model
    torch.save(net.state_dict(), dir_path + '/model_dict/' + str(config['algo']) +'-'+ idx_to_envname(config['expert']))

    for epoch in range(num_epochs):
        print("Starting epoch: {}".format(epoch))
        #do one pass of all of the batches in the dataloader and get back a histogram of losses across the batches
        loss_hist =loss_pass(net, loss_fn, RLDataloader, epoch, optimizer=optimizer)

        #find mean_loss and update graph
        loss_hist = np.asarray(loss_hist)
        mean_loss_item = np.mean(loss_hist)
        log_loss('Training Loss', mean_loss_item, epoch)

        #run model to get new reward
        returns, _, _ = run_best_policy()
        mean_reward = np.mean(returns)
        std_reward = np.std(returns)
        curr_reward = mean_reward - std_reward
        log_loss('Avg Reward', curr_reward, epoch)

        #update saved model
        if best_reward < curr_reward:
            print("New Best Reward! Saving..")
            torch.save(net.state_dict(), dir_path + '/model_dict/' + str(config['algo']) +'-'+ idx_to_envname(config['expert']))

def make_RLDataset(observations, expert_actions):
    expert_actions = torch.from_numpy(np.squeeze(expert_actions))
    observations = torch.from_numpy(observations)
    RLDataset = TensorDataset(observations, expert_actions)
    return RLDataset

def Dagger(net, RLDataset):
    dagger_iters = 20
    for i in range(dagger_iters):
        #first train on dataset
        train(net, RLDataset)

        #run best trained policy
        _, observations, actions = run_best_policy()

        #get expert policy on new observations
        expert_actions = get_expert_policy_fromobs(observations)
        # print("got fromobs" + dagger_iters)
        expert_actions = torch.from_numpy(np.squeeze(expert_actions))
        observations = torch.from_numpy(observations)
        RLDataset_new = TensorDataset(observations, expert_actions)
        RLDataset = ConcatDataset([RLDataset, RLDataset_new])

def Behavioural_Cloning(net, RLDataset):
    #straight train on dataset. nothing complicated
    train(net, RLDataset)

def main():
    global writer
    if config['run_train']:
        observations, expert_actions = get_expert_policy(config)
        input_dim = observations.shape[1]
        output_dim = expert_actions.shape[2]
        print("INPUT DIM:"  + str(input_dim))
        print("OUTPUT DIM:" + str(output_dim))
        
        #setup tensorboard stuff
        expert = config['expert']
        writer = SummaryWriter('runs/' + config['algo'] + idx_to_envname(expert))

        net = BaselineNN(input_dim, output_dim)
        net = net.float()
        #behavioural cloning or dagger
        RLDataset = make_RLDataset(observations, expert_actions)
        with tf.Session():
            tf_util.initialize()
            if(config['algo'] == 'b'):
                Behavioural_Cloning(net, RLDataset)
            else:
                Dagger(net, RLDataset)
        writer.close()
    run_best_policy()
if __name__ == "__main__":main()

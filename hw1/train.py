import os, sys
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

#Required for Tensorboard logging stuff
writer = None 

#using args is very verbose, so change parameters up here
config = {'expert':1, 'train_num_rollouts': 20, 'test_num_rollouts': 40,'run':True, 'algo':'b', 'epochs':10}
def initialize_net(env, env_name, algo):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    net = BaselineNN(input_dim, output_dim)
    net.load_state_dict(torch.load('/home/dhruvkar/Desktop/Robotics/deeprl/homework_pytorch/hw1/model_dict/'+ str(config['algo']) +'-'+ env_name))

def run_best_policy():
    expert_name = idx_to_envname(config['expert'])
    env = gym.make(expert_name)
    net = initialize_net(env, expert_name, config['algo'])

def idx_to_envname(idx):
    dir_path = os.getcwd() 
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
    if(config['run']):
        os.system('python run_expert.py experts/' + exp_policy + '.pkl ' + exp_policy +' --render --num_rollouts ' + str(num_rollouts))

    fn = expert_data_path + '/' + exp_policy + '.pkl'
    with open(fn, 'rb') as f:
        data = pickle.loads(f.read())
    return data['observations'], data['actions']

def log_loss(loss_item, epoch):
    global writer
    writer.add_scalar('Training Loss', loss_item, epoch)

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
    best_mean_loss = 0.0

    for epoch in range(num_epochs):
        print("Starting epoch: {}".format(epoch))
        #do one pass of all of the batches in the dataloader and get back a histogram of losses across the batches
        loss_hist =loss_pass(net, loss_fn, RLDataloader, epoch, optimizer=optimizer)

        #find mean_loss and update graph
        loss_hist = np.asarray(loss_hist)
        mean_loss_item = np.mean(loss_hist)
        log_loss(mean_loss_item, epoch)

        #update saved model
        if best_mean_loss < mean_loss_item:
            print("New Best Loss! Saving..")
            torch.save(net.state_dict(), '/home/dhruvkar/Desktop/Robotics/deeprl/homework_pytorch/hw1/model_dict/'+ str(config['algo']) +'-'+idx_to_envname(expert))

def make_RLDataset(observations, expert_actions):
    expert_actions_actions = torch.from_numpy(np.squeeze(expert_actions))
    observations = torch.from_numpy(observations)
    RLDataset = TensorDataset(observations, expert_actions)
    return RLDataset

def Dagger(net, RLDataset):
    dagger_iters = 20
    for i in range(dagger_iters):
        train(net, RLDataset)

def Behavioural_Cloning(net, RLDataset):
    #straight train on dataset and do regression. nothing complicated
    train(net, RLDataset)

def main():
    observations, expert_actions = get_expert_policy(config)
    global writer
    # pdb.set_trace()
    input_dim = observations.shape[1]
    output_dim = expert_actions.shape[2]
    print("INPUT DIM:"  + str(input_dim))
    print("OUTPUT DIM:" + str(output_dim))
    writer = SummaryWriter('runs/' + config['algo'] + idx_to_envname(expert))
    #
    net = BaselineNN(input_dim, output_dim)
    net = net.float()
    #behavioural cloning or dagger
    RLDataset = make_RLDataset(observations, expert_actions)
    if(config['algo'] == 'b'):
        Behavioural_Cloning(net, RLDataset)
    else:
        Dagger(net, RLDataset)
    writer.close()
if __name__ == "__main__":main()

"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Adapted for pytorch version by Ning Dai
"""
import numpy as np
import pdb
import torch
import gym
import logz
import argparse
import scipy.signal
import os
import time
import inspect
import math
from torch.multiprocessing import Process
from torch import nn, optim

class PolicyNet(nn.Module):
    def __init__(self, neural_network_args):
        super(PolicyNet, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['n_layers']

        self.build_model()

    def build_model(self):
        layer_dims = [self.ob_dim] + [self.hidden_size] * self.n_layers + [self.ac_dim]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.Tanh()])
        layers = layers[:-1]
        self.model = nn.Sequential(*layers).apply(self.weights_init_)
        if not self.discrete:
            self.ts_logsigma = nn.Parameter(torch.randn((self.ac_dim, )))
        
    def weights_init_(self, m):
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, ts_ob_no):
        """
        ts_ob_no: A Tensor with shape (batch_size * observation_dim)
        """
        y = self.model(ts_ob_no)
        if self.discrete:
            ts_logits_na = y
            return ts_logits_na
        else:
            ts_means_na = y
            ts_logsigma = self.ts_logsigma
            return (ts_means_na, ts_logsigma)

class Agent(object):
    def __init__(self, neural_network_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['n_layers']
        self.learning_rate = neural_network_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

        self.policy_net = PolicyNet(neural_network_args)
        params = list(self.policy_net.parameters())

        if self.nn_baseline:
            self.value_net = build_mlp(self.ob_dim, 1, self.n_layers, self.hidden_size)
            params += list(self.value_net.parameters())

        self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def sample_trajectories(self, itr, env):
        timesteps_this_batch = 0
        paths = []
        while timesteps_this_batch <= self.min_timesteps_per_batch:
            #Animate on first and every tenth batch 
            animate_this_episode = (len(paths) == 0 and (itr%10) == 0 and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate):
            ob = env.reset()
            obs, acs, rewards = [], [], []
            steps = 0
            while True:
                if animate:
                    env.render()
                    time.sleep(0.1)
                obs.append(ob)

                #==================
                #SAMPLE ACTION
                #===================
                #Convert to a tensor
                ts_ob_no = torch.from_numpy(ob[None]).float() #(1,ob_dim)
                #Pass through policynet
                if self.discrete:
                    ts_logits_na = self.policy_net(ts_ob_no)
                    ts_probs_na = nn.Softmax()(ts_logits_na)
                    ts_action = torch.multinomial(ts_probs_na, num_samples=1) #(1,1)
                    action_na = torch.squeeze(ts_action).numpy() # (1, )
                else:
                    ts_means_na, ts_logsigma_na = self.policy_net(ts_ob_no)
                    ts_action_na = torch.normal(ts_means_na, ts_logsigma_na.exp()) #(1, action_dim)
                    action_na = torch.squeeze(ts_action_na).numpy()
                acs.append(action_na)
                ob, rew, done, _ = env.step(action_na)
                rewards.append(rew)
                steps += 1
                if done or steps > self.max_path_length:
                    break
            path = {"observation":np.array(obs, dtype=np.float32), 
                    "reward":np.array(rewards, dtype=np.float32),
                    "action":np.array(acs, dtype=np.float32)}
            return path

    def estimate_return(self, obs_no, re_n):
        #==================
        #Transform re_n (num_episodes, each element is num_timesteps of that episode) to q_n (batchsize=sum_timesteps,)
        #===================
        sum_timesteps = obs_no.shape[0]
        q_n = np.array([])
        if self.reward_to_go:
            q_n = np.concatenate([scipy.signal.lfilter(b=[1], a=[1, -self.gamma], x=re[::-1])[::-1] for re in re_n]).astype(np.float32)
        else:
        #Transforms re_n into q_n where each index is the sum of rewards of the path is belonged to * self.gamma 
            q_n = np.concatenate([np.full_like(re, scipy.signal.lfilter(b=[1], a=[1, -self.gamma], x=re[::-1])[-1]) for re in re_n])
        #==================
        # Calculate  adv=(batchsize=sum_timesteps, ) by subtracting some baselines from estimated q_n=(batchsize=sum_timesteps, )
        #===================
        if self.nn_baseline:
            #Use Neural network baseline value function that predictions expected return conditioned on a state
            b_n = self.value_net(torch.from_numpy(obs_no)).view(-1).numpy()
            b_n = (b_n - np.mean(b_n)) / (np.std(b_n) + 1e-7)
            b_n = b_n * np.std(q_n) + np.mean(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-7)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        #convert numpy arrays to tensors
        ts_ob_no, ts_ac_na, ts_qn, ts_adv_n = map(lambda x: torch.from_numpy(x), [ob_no, ac_na, q_n, adv_n])

        #get Policy Distribution and log probabilities
        policy_params = self.policy_net(ts_ob_no)
        # pdb.set_trace()
        if self.discrete:
            ts_logits_na = policy_params
            ts_logprob_n = torch.distributions.Categorical(logits=ts_logits_na).log_prob(ts_ac_na)
        else:
            ts_means_na, ts_logsigma = policy_params
            ts_logprob_n = torch.distributions.Normal(ts_means_na, ts_logsigma.exp()).log_prob(ts_ac_na).sum(-1)

        #clean gradient and backprop mate
        self.optimizer.zero_grad()

        loss = -(ts_logprob_n * ts_adv_n).mean()
        loss.backward()
        if self.nn_baseline:
            baseline = self.value_net(ts_ob_no).view(-1)
            ts_target_n = (ts_qn - ts_qn.mean()) / (ts_qn.std() + 1e-7)
            baseline_loss = torch.nn.functional.mse_loss(baseline, ts_target_n)
            baseline_loss.backward()
        self.optimizer.step()
 

def train_PG(exp_name, env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        animate, 
        logdir, 
        normalize_advantages,
        nn_baseline, 
        seed,
        n_layers,
        size):
    start = time.time()
    #==================
    #SETUP LOGGER
    #===================
    locals_ = locals()
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    hyperparams = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(hyperparams)
 
    #==================
    #SETUP ENV
    #===================
    #Make gym env
    env = gym.make(env_name)
    #Set random seeds (TORCH, NUMPY and ENVIRONMENT)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    #Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    #Find out if env is continous or discrete
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    #Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    #==================
    #INITIALIZE AGENT
    #===================
    neural_network_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
    }
    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }
    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }
    agent = Agent(neural_network_args, sample_trajectory_args, estimate_return_args)

    #==================
    #TRAINING LOOP
    #===================
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        with torch.no_grad():
            #Step 1: Sample Trajectories from current policy (neural network)
            paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
        #Step 2: Calculate the RETURNS (Q_val, Adv) for this batch (batch_size = sum of all timesteps across all paths)
        ob_no = np.concatenate([path["observation"] for path in paths]) #(batch_size * obs_dim) 
        ac_na = np.concatenate([path["action"] for path in paths]) #(batch_size * action_dim)
        re_n = [path["reward"] for path in paths] #(num_paths) each index is a numpy array containing the rewards for that path
        with torch.no_grad():
            q_n, adv_n = agent.estimate_return(ob_no, re_n)
        #Step 3: Update parameters using Policy Gradient
        agent.update_parameters(ob_no, ac_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [len(path["reward"]) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.save_pytorch_model(agent)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # train_func()
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
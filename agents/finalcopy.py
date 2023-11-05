import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal
import os
from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
import torch.nn.functional as F



class ImitationAgent(BaseAgent):

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(5000) 
        self.beta = 0.5
        self.cur_max_reward = 0
        self.save = self.hyperparameters["save"]

        self.model = nn.Sequential(
                nn.LayerNorm(self.observation_dim),
                nn.Linear(self.observation_dim, 4*self.observation_dim),
                nn.LeakyReLU(),
                nn.Linear(4*self.observation_dim, 2*self.observation_dim),
                nn.LayerNorm(2*self.observation_dim),
                nn.Linear(2*self.observation_dim, self.action_dim),
                nn.Tanh()
            )
        self.learning_rate = 1e-3
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def forward(self, observation: torch.FloatTensor):
        return self.model(observation)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        return self.model(observation)
    
    
    def update(self, trajs, env):
        j = 0
        for traj in trajs:
            for i in range(len(traj["action"])):
                obsvn = torch.from_numpy(traj["observation"][i])
                if random.random() < self.beta:
                    ac = torch.from_numpy(self.expert_policy.get_action(torch.from_numpy(traj["observation"][i])))
                else:
                    ac = torch.from_numpy(traj["action"][i])
                y_pred = self.model(obsvn)
                loss = self.loss_fn(y_pred, ac)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
               
        
        for _ in range(10):
            rand_int = random.randint(0, len(self.replay_buffer.paths)-1)
            replay_traj = self.replay_buffer.paths[rand_int]
            for i in range(len(replay_traj["reward"])):

                obsvn = torch.from_numpy(replay_traj["observation"][i])
                if random.random() < self.beta:
                    ac = torch.from_numpy(self.expert_policy.get_action(torch.from_numpy(replay_traj["observation"][i])))
                else:
                    ac = torch.from_numpy(replay_traj["action"][i])
                y_pred = self.model(obsvn)
                loss = self.loss_fn(y_pred, ac)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

           
        self.replay_buffer.add_rollouts(trajs)
        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        upd = np.mean(eval_returns)
        return upd
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)

        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False) #
        self.beta = 1 / (1 + envsteps_so_far/1000)
        upd = self.update(trajs, env)
        print("Model score : ",  upd)
        if self.save and  upd > self.cur_max_reward:
            print("Saving model with score : ",  upd)
            self.cur_max_reward = upd
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))
        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} 
import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(5000) #you can set the max size of replay buffer if you want
        self.beta = 0.5

        #initialize your model and optimizer and other variables you may need
        

        self.model = nn.Sequential(
                #nn.Conv1d(1, 12, 3),
                #nn.Conv1d(12, 1, observation_dim-action_dim-1)
                nn.Linear(self.observation_dim, 20),
                nn.LeakyReLU(),
                nn.Linear(20, self.action_dim),
                nn.LeakyReLU()
            )
        self.learning_rate = 1e-3
        self.loss_fn = nn.MSELoss(reduction='sum')
        #self.conv1 = nn.Conv1d(1, 12, 3)
        #self.conv2 = nn.Conv1d(12, 1, observation_dim-action_dim-1)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        #print("required shape = ", self.action_dim)
        #print("actual shape = ", self.replay_buffer[0].shape)
        #action = torch.from_numpy(self.replay_buffer.acs[0]) #change this to your action

        # x = nn.functional.relu(self.conv1(observation))
        # x = nn.functional.relu(self.conv2(x))
        # #print("Forwading action shape =", x.shape)
        # return x
    
        # action = torch.from_numpy(self.expert_policy.get_action(observation))
        # return action
        return self.model(observation)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        
        #action = torch.from_numpy(self.replay_buffer.acs[0]) #change this to your action
        # x = nn.functional.relu(self.conv1(observation))
        # #print("Forwading action shape1 =", x.shape)
        # x = nn.functional.relu(self.conv2(x))
        # #print("Forwading action shape2 =", x.shape)
        # return x
    
        # action = torch.from_numpy(self.expert_policy.get_action(observation))
        # return action 
        return self.model(observation)
    
    
    def update(self, trajs):
        #*********YOUR CODE HERE******************
        avg_loss = 0.0
        j = 0
        for traj in trajs:
            if len(traj["action"])==len(traj["observation"]):
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
                    avg_loss += loss.item()
                    j += 1
            else:
                print("oops?")
        
        self.replay_buffer.add_rollouts(trajs)
        return avg_loss/j
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
        
        #print("buffer number of paths = ", len(self.replay_buffer.paths))
        #print("buffer first traj obs, reward, next_obs, action, terminal= ", len(self.replay_buffer.paths[0]["observation"]), len(self.replay_buffer.paths[0]["reward"]), len(self.replay_buffer.paths[0]["next_observation"]), len(self.replay_buffer.paths[0]["action"]), len(self.replay_buffer.paths[0]["terminal"]))
        #print("buffer first obs, ac shape= ", self.replay_buffer.obs[0].shape, self.replay_buffer.acs[0].shape)
        #print("buffer first obs, ac = ", self.replay_buffer.obs[0], self.replay_buffer.acs[0])
        
        #print("actual shape = ", torch.from_numpy(self.replay_buffer.acs[0]).shape)
        #*********YOUR CODE HERE******************
        #for i in range(len(envsteps_so_far, min(envsteps_so_far+10, len(self.replay_buffer.obs)))):
        #print("env slice = ",envsteps_so_far ,  min(envsteps_so_far+10, len(self.replay_buffer.obs)) )

        max_lengths = list(map(len, self.replay_buffer.paths))
        # Find the maximum length
        print("min, avg max of traj lengths orig", max(max_lengths), np.average(max_lengths), max(max_lengths))
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False) #
        #print("N = ", len(trajs), " number of trajectories sampled")
        #print("These trajs are ", trajs[0])
        # obsvns = []
        # acns = []
        # for traj in trajs:
        #     obsvns.append(traj["observation"])
        #     acns.append(traj["action"])


        #obsvns1 = np.concatenate(self.replay_buffer.obs, trajs['observations']) #[envsteps_so_far : min(envsteps_so_far+10, len(self.replay_buffer.obs))]
        #acns1 = self.replay_buffer.acs#[envsteps_so_far : min(envsteps_so_far+10, len(self.replay_buffer.obs))]
        #upd = self.update(obsvns, acns)
        self.beta = 1 / (1 + envsteps_so_far/1000)
        upd = self.update(trajs)
        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': 50} #you can return more metadata if you want to









class RLAgent(BaseAgent):

    '''
    Please implement an policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need
        self.model = nn.Sequential(
                nn.Linear(self.observation_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 4*self.action_dim),
                nn.LeakyReLU(),
                nn.Linear(4*self.action_dim, 2*self.action_dim),
            )
       
        self.loss_fn = nn.GaussianNLLLoss()
        #self.loss_fn = nn.MSELoss(reduction = 'sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())

    
    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        x = self.model(observation)
        a_m = x[:, :self.action_dim]
        a_v = torch.clamp(torch.abs(x[:,self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        return action
    
    
    def distbn(self, observation: torch.FloatTensor):
        x = self.model(observation)
        a_m = x[:self.action_dim]
        a_v = torch.clamp(torch.abs(x[self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        log_action = ms.log_prob(action)
        return action, log_action, a_m, a_v


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        x = self.model(observation)
        a_m = x[:,:self.action_dim]
        a_v = torch.clamp(torch.abs(x[:,self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        return action

    
    '''def update(self, observations, actions, advantage, q_values = None):
        #*********YOUR CODE HERE******************
        loss = self.policy_update(observations, actions, advantage)
        if self.hyperparameters['critic']:
                critic_loss = self.critic_update(observations, q_values)
                loss += critic_loss
        return loss

        return'''
    
    def update(self, trajs):
        avg_score = 0.0
        n = len(trajs)
        losses = []
        for traj in trajs:
            # rewards = []
            # log_actions = []
            score = 0
            li = []
            rewards = []
            for step in range(len(traj["observation"])):
                #select action, clip action to be [-1, 1]
                state = torch.from_numpy(traj["observation"][step])
                action_suggested, la, a_m, a_v = self.distbn(state) #clip action? action = min(max(-1, action), 1)
                actual_action = torch.from_numpy(traj["action"][step])
                #l1 = torch.norm(la)
                l1 = self.loss_fn(actual_action, a_m, a_v)
                # log_probs = policy_network.log_prob(actions, means, log_stds)
                # l1 = -torch.mean(log_probs * rewards)
                
                reward = traj['reward'][step]
                score += reward * (self.hyperparameters["gamma"]**step) #track episode score
                li.append(l1)
                rewards.append(reward)

            acc_rewards = []
            pr = 0.0
            for r in rewards[::-1]:
                next_r = r + self.hyperparameters["gamma"]*pr
                acc_rewards.append(next_r)
                pr = next_r
            acc_rewards.reverse()
            #print("rewards len = ", len(rewards), " and acc rewards len = ", len(acc_rewards))
            #print("acc rews =", acc_rewards[0], acc_rewards[1], acc_rewards[2])
            for step in range(len(li)):
                li[step] = acc_rewards[step] * li[step]

            losses.append(sum(li))   
            #Calculate Gt (cumulative discounted rewards)
            #rewards = process_rewards(rewards)
            #adjusting policy parameters with gradient ascent
            # loss = []
            # for r, la in zip(rewards, log_actions):
                
            #     loss.append(-r * la) # a negative sign since network will perform gradient descent and we are doing gradient ascent
        loss = (1/n) * sum(losses)    
        #Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        avg_score += score

        return avg_score/n
    

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        self.train()
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        upd = self.update(trajs)

        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to


















class ImitationSeededRL(ImitationAgent, RLAgent):
    '''
    Implement a policy gradient agent with imitation learning initialization.
    You can use the ImitationAgent and RLAgent classes as parent classes.

    Note: We will evaluate the performance on Ant domain only. 
    If everything goes well, you might see an ant running and jumping as seen in lecture slides.
    '''
    
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need






    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        #you may want to use the itr_num to decide between IL and RL
        self.train()
        
        return {'episode_loss': None, 'trajectories': None, 'current_train_envsteps': None} #you can return more metadata if you want to




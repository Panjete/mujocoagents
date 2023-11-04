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

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(5000) #you can set the max size of replay buffer if you want
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
    
    
    def update(self, trajs):
        avg_loss = 0.0
        avg_score = 0.0
        j = 0
        for traj in trajs:
            for i in range(len(traj["action"])):
                obsvn = torch.from_numpy(traj["observation"][i])
                avg_score += traj["reward"][i]
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
           
        
        self.replay_buffer.add_rollouts(trajs)
        return avg_score/len(trajs)
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)

        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False) #
        self.beta = 1 / (1 + envsteps_so_far/1000)
        upd = self.update(trajs)
        if self.save and itr_num%10 == 0 and  upd > self.cur_max_reward:
            print("saving Model with score : ",  upd)
            self.cur_max_reward = upd
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))
        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to

##########################
##########################
##########################
##########################


# Self trained Actor critic
class RLAgent(BaseAgent):
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
       

        self.actor = nn.Sequential(
                nn.LayerNorm(self.observation_dim),
                nn.Linear(self.observation_dim, 4*self.observation_dim),
                nn.LeakyReLU(),
                nn.Linear(4*self.observation_dim, 4*self.action_dim),
                nn.LayerNorm(4*self.action_dim),
                nn.Linear(4*self.action_dim, 2*self.action_dim),
                nn.Tanh()
            )
    
        self.critic = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, 4*self.action_dim),
            nn.LeakyReLU(),
            nn.Linear(4*self.action_dim, 2*self.action_dim),
            nn.LeakyReLU(),
            nn.Linear(2*self.action_dim, 1),
        )
    
        self.loss_fn = nn.GaussianNLLLoss()
        self.loss_critic = nn.MSELoss(reduction = 'sum')
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
        self.sampling = False
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, observation: torch.FloatTensor):
        x = self.actor(observation)
        a_m = x[:,:self.action_dim]
        a_v = torch.clamp(torch.abs(x[:,self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        return action

    def forward_(self, observation: torch.FloatTensor):
        x = self.actor(observation)
        a_m = x[:self.action_dim]
        a_v = torch.clamp(torch.abs(x[self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        return action
    
    def distbn(self, observation: torch.FloatTensor):
        x = self.actor(observation)
        a_m = x[:self.action_dim]
        a_v = torch.clamp(torch.abs(x[self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        log_action = ms.log_prob(action)
        return action, log_action, a_m, a_v


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        x = self.actor(observation)
        a_m = x[:,:self.action_dim]
        a_v = torch.clamp(torch.abs(x[:,self.action_dim:]), min = self.hyperparameters["std_min"])
        ms = torch.distributions.Normal(a_m, a_v)
        action = ms.sample()
        r = random.random()
        # if r > self.hyperparameters["prob_rand_sample_training"]: ## Happens with v little prob 
        #     action = 2 * torch.rand((1, self.action_dim)) - 1
        return action

    def update_baseline(self, trajs):
        self.train()
        avg_score = 0.0
        n = len(trajs)
        losses = []
        for traj in trajs:
            ## Step 1: getting trajectories
            score = 0
            li = [] ## Houses log(pi_theta(a_it|s_it)) for all t in this traj i
            rewards = []
            traj_len = len(traj["observation"])
            for step in range(len(traj["observation"])):
                #select action, clip action to be [-1, 1]
                state = torch.from_numpy(traj["observation"][step])
                action_suggested, la, a_m, a_v = self.distbn(state) #clip action? action = min(max(-1, action), 1)
                actual_action = torch.from_numpy(traj["action"][step])
                #l1 = torch.norm(la)
                l1 = self.loss_fn(actual_action, a_m, a_v)
                reward = traj['reward'][step] + 0.2 * (((traj_len - step) ** 2)) # Highly promoting higher epsiode lengths
                score += reward * (self.hyperparameters["gamma"]**step) #track episode score
                li.append(l1)
                rewards.append(reward)

            avg_reward = sum(rewards)/len(rewards)
            for step in range(len(li)):
                ## Step 3 : evaluating A_pi
                reward_this_time = traj["reward"][step]
                A_pi_s_i = reward_this_time-avg_reward
                ## Step 4 : log(pi(a_it, s_it)) * A_pi_s_i
                li[step] = li[step] * A_pi_s_i
                #li[step] = acc_rewards[step] * li[step] ## Without baseline, log(pi(a_it, s_it)) * sum(r(s_t', a_t')) for t' = t to T
            
            losses.append(sum(li))

        loss = self.hyperparameters["alpha"] * (1/n) * sum(losses)    
        #Step 5 : Backpropagation on theta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        avg_score += score

        return avg_score/n
    
    def update_ac(self, trajs):
        self.train()
        avg_score = 0.0
        n = len(trajs)
        losses = []
        for traj in trajs:
            ## Step 1: getting trajectories
            score = 0
            li = [] ## Houses log(pi_theta(a_it|s_it)) for all t in this traj i
            rewards = []
            traj_len = len(traj["observation"])

            ## Accumulate rewards
            for step in range(len(traj["observation"])):
                reward = traj['reward'][step] + 0.2 * (((traj_len - step) ** 2)) # Highly promoting higher epsiode lengths
                score += reward * (self.hyperparameters["gamma"]**step) #track episode score
                rewards.append(reward)

            ## This loop does gives the "rewards to go" from every (s, a) in this trajectory
            acc_rewards = []
            pr = 0.0
            for r in rewards[::-1]:
                next_r = r + self.hyperparameters["gamma"]*pr
                acc_rewards.append(next_r)
                pr = next_r
            acc_rewards.reverse()
                
            
            for step in range(len(acc_rewards)):
                ## Step 2. Fit V_phi_pi to sampled new rewards
                trj_reward_discounted = torch.tensor([acc_rewards[step]], dtype = torch.float).to(self.device)
                state = torch.tensor(traj["observation"][step]).to(self.device)
                estimated_reward = self.critic(state)
                critic_loss = self.loss_critic(trj_reward_discounted, estimated_reward)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            for step in range(len(li)):
                ## Step 3 : evaluating A_pi
                s_prime = torch.tensor(traj["next_observation"][step], dtype = torch.float).to(self.device)
                V_s_prime = self.critic(s_prime)
                state = torch.tensor(traj["observation"][step], dtype = torch.float).to(self.device)
                V_state = self.critic(state)
                reward_this_time = traj["reward"][step]
                A_pi_s_i = reward_this_time + (self.hyperparameters["gamma"]*V_s_prime) - V_state

                action_suggested, la, a_m, a_v = self.distbn(state)
                actual_action = torch.tensor(traj["action"][step], dtype = torch.float).to(self.device)
                loss_unweighted = acc_rewards[step] * self.loss_fn(actual_action, a_m, a_v)
                loss_weighted = self.hyperparameters["alpha"] * (1/n)  * loss_unweighted * A_pi_s_i
                self.optimizer_actor.zero_grad()
                loss_weighted.backward()
                self.optimizer_actor.zero_grad()
                #li[step] = acc_rewards[step] * li[step] ## Without baseline, log(pi(a_it, s_it)) * sum(r(s_t', a_t')) for t' = t to T

            avg_score += score

        return avg_score/n

    def learn_actor(self, env, itr_nums, ntraj, maxtraj, imitator:ImitationAgent):
        self.train()
        print("Training Actor")
        for _ in range(itr_nums):
            trajs = utils.sample_n_trajectories(env, self, ntraj, maxtraj, False)
            for traj in trajs:
                for step in range(len(traj["observation"])):
                    state = torch.tensor(traj["observation"][step], dtype = torch.float, requires_grad=True).to(self.device) ## s - the current states being analysed
                    action = torch.tensor(traj["action"][step], dtype = torch.float, requires_grad=True).to(self.device) ## action a actually taken in the trajectory
                    actions_to_replicate = torch.tensor(imitator.get_action(state), dtype = torch.float, requires_grad=True)

                    #print("Actions DIM : ", action.size(), " reqd action DIM : ", actions_to_replicate.size())
                    actions = self.forward_(state)  ## Find actions and corres. log_prob suggested by actor network, Do not reparametrize
        
                    actor_loss = F.mse_loss(actions, actions_to_replicate, reduction="mean")
                    #print(actor_loss)
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self.optimizer_actor.step()
        return
    
    def learn_critics(self, env, itr_nums, ntraj, maxtraj):
        self.train()
        print("Training Critic")
        for _ in range(itr_nums):
            trajs = utils.sample_n_trajectories(env, self, ntraj, maxtraj, False)
            for traj in trajs:
                rewards = traj["reward"]
                state = torch.tensor(traj["observation"], dtype = torch.float).to(self.device) ## s - the current states being analyse
        
                acc_rewards = []
                pr = 0.0
                for r in rewards[::-1]:
                    next_r = r + self.hyperparameters["gamma"]*pr
                    acc_rewards.append(next_r)
                    pr = next_r
                acc_rewards.reverse()

                acc_rewards = torch.tensor(acc_rewards, dtype = torch.float).to(self.device) ## Rewards earned in this trajectory

                self.optimizer_critic.zero_grad()
                q_hat = acc_rewards
                q_op = self.critic.forward(state).view(-1) ## How good is the suggested state pair
                critic_loss = F.mse_loss(q_op, q_hat)
                critic_loss.backward()
                self.optimizer_critic.step()

                
        return
    

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        self.train()
        self.sampling = True
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        self.sampling = False
        self.hyperparameters["alpha"] = 1/(1 + envsteps_so_far/1000)
        upd = self.update_ac(trajs)
        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to
    


##########################
##########################
##########################
##########################


class ImitationSeededRL(ImitationAgent, RLAgent):
    
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters):
        RLAgent.__init__(self, observation_dim=observation_dim, action_dim=action_dim, args=args, discrete=discrete, **hyperparameters)
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.imitator = ImitationAgent(observation_dim, action_dim, args, discrete, **hyperparameters)
        self.imitator_trained = False

        self.rlagent = RLAgent(observation_dim, action_dim, args, discrete, **hyperparameters)
        self.cur_max_reward = 0.0

    def forward(self, observation: torch.FloatTensor):
        actions = self.rlagent.get_action(observation)
        return actions
    
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        actions = self.rlagent.get_action(observation)
        return actions

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        self.train()
        if self.imitator_trained == False:
            ## One time thing only
            total_envsteps = 0
            for itr in range(self.hyperparameters["imitator_itr"]):
                print(f"\n********** Imitation Phase Iteration {itr} ************")
                train_info = self.imitator.train_iteration(env, envsteps_so_far = total_envsteps, render=False, itr_num = itr)
                total_envsteps += train_info['current_train_envsteps']
            print("Imitator Initialised and Trained successfully")
            self.rlagent.learn_actor(env, self.hyperparameters["actor_learn_itr"], self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], self.imitator)
            print("RL Actor trained and Initialised successfully")
            self.rlagent.learn_critics(env, self.hyperparameters["critic_learn_itr"], self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"])
            print("RL Critics trained and Initialised successfully")
            self.imitator_trained = True
            
        fraction_done = (itr_num/self.hyperparameters["total_train_iterations"])**(0.8)
        r = random.random()
        if fraction_done < r:
            ## Happens less and less frequently as time passes
            ## train Imitator
            print("Seeded Model Re-Calibrating with Imitation Agent for itr = ", itr_num)
            self.imitator.train_iteration(env, envsteps_so_far = 10000)
            self.rlagent.learn_actor(env, self.hyperparameters["recalibrating_iters"], self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], self.imitator)
            self.rlagent.learn_critics(env,self.hyperparameters["recalibrating_iters"], self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"])
        else:
            ## Happens more and more frequently as time passes
            ## ALlow model to explore
            print("Seeded Model Exploring for itr = ", itr_num)
            for _ in range(self.hyperparameters["exploring_iters"]):
                trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
                self.rlagent.update_ac(trajs)

        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        rewards = [sum(traj["reward"]) for traj in trajs]
        cur_reward = sum(rewards)/len(rewards)
        print("Reward in this iteration of training = ", cur_reward)
        if  cur_reward > self.cur_max_reward:
            print("Saving model with avg score = ", cur_reward)
            self.cur_max_reward = cur_reward
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+"_ACself.pth"))
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        return {'episode_loss': 0.0, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to
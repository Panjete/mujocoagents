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


###
###
###
# This is our implementation of the Soft Actor Critic proposed in : https://arxiv.org/abs/1801.01290
# Materialisation of the concepts presented in the paper with the help of : https://www.youtube.com/watch?v=ioidsRlf79o
# Kindly also find attached our implementation of the vanilla actor-critic & baseline approach at the end of the code
# If the testing of these is required, kindly simply uncomment them and comment these 
# Since the ImitationSeededRL implementation also depends on the implementation, we had to tailor it to SAC as well
# The ImitationSeededRL implementation is purely our own, for both SAC and AC
###
###
###


# ''' <<<<<<<<<<<<<<<< uncomment this comment initiator to comment SAC, also for the closing block comment


# For calculating goodness of State Action Pair
class Critic(nn.Module):
    def __init__(self, observation_dim:int, action_dim:int, beta_critic, name):
        super(Critic, self).__init__()
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.name = name
        

        #initialize your model and optimizer and other variables you may need
        
        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, 256)
        self.fc2 = nn.Linear(256 , 256)
        self.q_out = nn.Linear(256 , 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta_critic)
        
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(torch.cat([state, action], dim = 1))
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        q_out = self.q_out(x)

        return q_out
    
## For calculating goodness of State
class Value(nn.Module):
    def __init__(self, observation_dim:int, beta_value, name):
        super(Value, self).__init__()
        self.observation_dim = observation_dim
        self.name = name
        self.lr = beta_value

        #initialize your model and optimizer and other variables you may need
    
        self.fc1 = nn.Linear(self.observation_dim, 256)
        self.fc2 = nn.Linear(256 , 256)
        self.v_out = nn.Linear(256 , 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        q_out = self.v_out(x)

        return q_out

## For state -> action prediction
class Actor(nn.Module):
    def __init__(self, alpha, observation_dim:int, action_dims, name):
        super(Actor, self).__init__()
        self.observation_dim = observation_dim
        self.name = name
        self.action_dims = action_dims
        self.noise_variance = 1e-4

        self.fc1 = nn.Linear(self.observation_dim, 256)
        self.fc2 = nn.Linear(256 , 256)

        self.means = nn.Linear(256 , self.action_dims)
        self.variances = nn.Linear(256, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
       
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, observation):

        x = self.fc1(observation)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        means = self.means(x) ## Mean of the distribution
        variances = torch.clamp(self.variances(x), min = self.noise_variance, max = 1) ## To ensure variances are positive, and bounded
        return means, variances
    
    def sample_normal(self, observation, r):
        means, variances = self.forward(observation)
        distbn = Normal(means, variances)
        if r:
            actions = distbn.rsample() ## Introduces extra randomness -> Use at training time to expedite exploration
        else:
            actions = distbn.sample()
        
        action = torch.tanh(actions).to(self.device)
        log_probs = distbn.log_prob(actions) - torch.log( 1- action.pow(2) + self.noise_variance) ## Also incorporate Entropy of the actions
        log_probs = log_probs.sum(1, keepdim = True)

        return action, log_probs

#### SAC based 
class RLAgent(BaseAgent):

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args

        # Initializing hyper-parameters

        self.alpha = self.hyperparameters["alpha"]   ## Learning Rate of the Actor Network
        self.beta = self.hyperparameters["beta"]     ## Learning rates of the value estimators and critics
        self.gamma = self.hyperparameters["gamma"]    ## Discounting Rate
        self.tau = self.hyperparameters["tau"]        ## Soft-copy deciding parameter
        self.batch_size = self.hyperparameters["ntraj"]   ## Batch Size, though code we do a single trajectory anyways
        self.reward_scale = self.hyperparameters["reward_scale"]    ## For determining what weightage to give to reward and predicted reward

        # Initialising predictors and quality estimators
        #print("MAX TRAJ LEN =", self.hyperparameters["maxtraj"])
        self.actor = Actor(self.alpha, self.observation_dim, self.action_dim, "Actor")   
        self.critic1 = Critic(self.observation_dim, self.action_dim, self.beta, "Critic1")
        self.critic2 = Critic(self.observation_dim, self.action_dim, self.beta, "Critic2")

        self.value =  Value(self.observation_dim, self.beta, "Value")
        self.target_value =  Value(self.observation_dim, self.beta, "target_value") ## one for updates, one for soft transition

        self.cur_max_reward = 0.0
        self.update_network_parameters(tau = 1) ##initialising value's parameters to be the same as target_value
       
        
    def forward(self, observation: torch.FloatTensor):
        actions, _ = self.actor.sample_normal(observation, r = False)
        return actions

    # For soft transition of the model's parameter
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
            ## soft copy hard copy mechanism
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        tvs_dict = dict(target_value_params)
        vs_dict = dict(value_params)

        for name in vs_dict:
            vs_dict[name] = tau * vs_dict[name].clone() + (1-tau) * tvs_dict[name].clone()

        self.target_value.load_state_dict(vs_dict)

    ## Filhaal learns one trajectory at a time, maybe try to implement a batch one?
    def learn(self, trajs, env):
        self.train()
        rewards_earned = 0.0
        for traj in trajs:
            reward = torch.tensor(traj["reward"], dtype = torch.float).to(self.actor.device) ## Rewards earned in this trajectory
            state_ = torch.tensor(traj["next_observation"], dtype = torch.float).to(self.actor.device) ## s' - the states the trajectory goes into after action a
            state = torch.tensor(traj["observation"], dtype = torch.float).to(self.actor.device) ## s - the current states being analysed
            action = torch.tensor(traj["action"], dtype = torch.float).to(self.actor.device) ## action a actually taken in the trajectory
            done = torch.tensor(traj["terminal"], dtype = torch.float).to(self.actor.device) ## Terminal state or not?

            #np.concatenate(numpy_arrays, axis=0)
            value = self.value(state).view(-1) ## Collapse across batch dimension - anyways 1
            value_ = self.target_value(state_).view(-1) ## Collapse across batch dimension - anyways 1
            value_[-1] = 0.0

            actions, log_probs = self.actor.sample_normal(state, r = False) ## Find actions and corres. log_prob suggested by actor network, Do not reparametrize
            #log_probs = log_probs#.view(-1)
            q1_np = self.critic1.forward(state,actions) ## How good is the suggested state-action pair
            q2_np = self.critic2.forward(state,actions) ## How good is the suggested state-action pair, second opinion 
            critic_value = torch.min(q1_np, q2_np) ## For removing over-estimation bias
            critic_value = critic_value.view(-1) 

            self.value.optimizer.zero_grad()
            value_target = critic_value - log_probs.view(-1) ## 
            value_loss = 0.5 * F.mse_loss(value, value_target) ## How far is the estimation of 
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            actions, log_probs = self.actor.sample_normal(state, r = True)
            log_probs = log_probs.view(-1)
            q1_np = self.critic1.forward(state,actions)
            q2_np = self.critic2.forward(state,actions)
            critic_value = torch.min(q1_np, q2_np)
            critic_value = critic_value.view(-1)

            actor_loss = torch.mean(log_probs- critic_value)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            q_hat = self.reward_scale * reward + self.gamma * value_
            q1_op = self.critic1.forward(state,action).view(-1)
            q2_op = self.critic2.forward(state,action).view(-1)
            critic_loss = 0.5 * (F.mse_loss(q1_op, q_hat) + F.mse_loss(q2_op, q_hat))
            critic_loss.backward()
            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            self.update_network_parameters()

            rewards_earned += float(reward.sum())
        #print("REWARDS EARNED IN THIS LEARNING CYCLE =", rewards_earned)
        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        upd = np.mean(eval_returns)
        return  upd
        
        
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        actions, _ = self.actor.sample_normal(observation, r = False)
        return actions

    def learn_actor(self, env, itr_nums, ntraj, maxtraj, imitator:ImitationAgent):
        self.train()
        print("Training Actor")
        for _ in range(itr_nums):
            trajs = utils.sample_n_trajectories(env, self, ntraj, maxtraj, False)
            for traj in trajs:
                
                state = torch.tensor(traj["observation"], dtype = torch.float).to(self.actor.device) ## s - the current states being analysed
                action = torch.tensor(traj["action"], dtype = torch.float).to(self.actor.device) ## action a actually taken in the trajectory
                actions_to_replicate = imitator.forward(state)

                #print("Actions DIM : ", action.size(), " reqd action DIM : ", actions_to_replicate.size())
                actions, _ = self.actor.sample_normal(state, r = True)  ## Find actions and corres. log_prob suggested by actor network, Do not reparametrize
    
                actor_loss = F.mse_loss(actions, actions_to_replicate, reduction="mean")
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.update_network_parameters()
        return
    
    def learn_critics(self, env, itr_nums, ntraj, maxtraj):
        self.train()
        print("Training Critics")
        for _ in range(itr_nums):
            trajs = utils.sample_n_trajectories(env, self, ntraj, maxtraj, False)
            for traj in trajs:
                reward = torch.tensor(traj["reward"], dtype = torch.float).to(self.actor.device) ## Rewards earned in this trajectory
                state_ = torch.tensor(traj["next_observation"], dtype = torch.float).to(self.actor.device) ## s' - the states the trajectory goes into after action a
                state = torch.tensor(traj["observation"], dtype = torch.float).to(self.actor.device) ## s - the current states being analysed
                action = torch.tensor(traj["action"], dtype = torch.float).to(self.actor.device) ## action a actually taken in the trajectory
                done = torch.tensor(traj["terminal"], dtype = torch.float).to(self.actor.device) ## Terminal state or not?

                #np.concatenate(numpy_arrays, axis=0)
                value = self.value(state).view(-1) ## Collapse across batch dimension - anyways 1
                value_ = self.target_value(state_).view(-1) ## Collapse across batch dimension - anyways 1
                value_[-1] = 0.0

                actions, log_probs = self.actor.sample_normal(state, r = False) ## Find actions and corres. log_prob suggested by actor network, Do not reparametrize
                #log_probs = log_probs#.view(-1)
                q1_np = self.critic1.forward(state,actions) ## How good is the suggested state-action pair
                q2_np = self.critic2.forward(state,actions) ## How good is the suggested state-action pair, second opinion 
                critic_value = torch.min(q1_np, q2_np) ## For removing over-estimation bias
                critic_value = critic_value.view(-1) 

                self.value.optimizer.zero_grad()
                value_target = critic_value - log_probs.view(-1) ## 
                value_loss = 0.5 * F.mse_loss(value, value_target) ## How far is the estimation of 
                value_loss.backward(retain_graph=True)
                self.value.optimizer.step()


                self.critic1.optimizer.zero_grad()
                self.critic2.optimizer.zero_grad()
                q_hat = self.reward_scale * reward + self.gamma * value_
                q1_op = self.critic1.forward(state,action).view(-1)
                q2_op = self.critic2.forward(state,action).view(-1)
                critic_loss = 0.5 * (F.mse_loss(q1_op, q_hat) + F.mse_loss(q2_op, q_hat))
                critic_loss.backward()
                self.critic1.optimizer.step()
                self.critic2.optimizer.step()

                self.update_network_parameters()
        return
    
    # def update(self, observations, actions, advantage, q_values = None):
    #     #*********YOUR CODE HERE******************
    #     loss = self.policy_update(observations, actions, advantage)
    #     if self.hyperparameters['critic']:
    #             critic_loss = self.critic_update(observations, q_values)
    #             loss += critic_loss
    #     return loss

    #     return
    

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        self.train()
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        cur_reward = self.learn(trajs, env)
        print("Models reward: ", cur_reward)
        if cur_reward > self.cur_max_reward:
            print("Saving model with score", cur_reward)
            self.cur_max_reward = cur_reward
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))
        return {'episode_loss': cur_reward, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to

    
### SAC based
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
        actions, _ = self.rlagent.actor.sample_normal(observation, r = False)
        return actions
    
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        actions, _ = self.rlagent.actor.sample_normal(observation, r = False)
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
            
        fraction_done = (1+ itr_num)/(itr_num+10)
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
                self.rlagent.learn(trajs, env)

        max_ep_len = env.spec.max_episode_steps
        trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in trajs]
        cur_reward = np.mean(eval_returns)
        print("Reward in this iteration of training ImitationSeededRL = ", cur_reward)
        if envsteps_so_far%100 == 0 and cur_reward > self.cur_max_reward:
            print("Saving model with avg score = ", cur_reward)
            self.cur_max_reward = cur_reward
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        return {'episode_loss': 0.0, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to

        

# ''' <<<<<<<<<<<<<<<< uncomment this comment to comment SAC





# Uncomment below block to run AC RL and AC-ImitationSeeded RL        

'''
class RLAgent(BaseAgent):
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(50000)
       
        self.model = nn.Sequential(
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
    
        self.loss_fn = nn.GaussianNLLLoss() ## Computes the Negative Log Likelihood - something we'd like to minimise, just like a loss!
        self.loss_critic = nn.MSELoss(reduction = 'sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters())
        self.sampling = False
        self.actor_critic = self.hyperparameters["actor_critic"]
        self.cur_max_reward = 0.0
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, observation: torch.FloatTensor):
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
        r = random.random()
        if self.sampling and r > self.hyperparameters["prob_rand_sample_training"]: ## Happens with v little prob 
            action = 2 * torch.rand((1, self.action_dim)) - 1
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
    
    def update_ac(self, trajs, env):
        self.train()
        if len(self.replay_buffer.paths) > 100:
            for _ in range(20):
                rand_int = random.randint(0, len(self.replay_buffer.paths)-1)
                replay_traj = self.replay_buffer.paths[rand_int]
                trajs.append(replay_traj)
        n = len(trajs)
        losses = []
        for traj in trajs:
            ## Step 1: getting trajectories
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
                li.append(l1)
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
                trj_reward_discounted = torch.Tensor([acc_rewards[step]]).to(torch.float64)
                state = torch.from_numpy(traj["observation"][step])
                estimated_reward = self.critic(state).to(torch.float64)
                critic_loss = self.loss_critic(trj_reward_discounted, estimated_reward).to(torch.float64)       
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            for step in range(len(li)):
                ## Step 3 : evaluating A_pi
                s_prime = torch.from_numpy(traj["next_observation"][step])
                V_s_prime = self.critic(s_prime).item()
                state = torch.from_numpy(traj["observation"][step])
                V_state = self.critic(state).item()
                reward_this_time = traj["reward"][step]
                A_pi_s_i = reward_this_time + (self.hyperparameters["gamma"]*V_s_prime) - V_state
                ## Step 4 : log(pi(a_it, s_it)) * A_pi_s_i
                li[step] = li[step] * A_pi_s_i
                #li[step] = acc_rewards[step] * li[step] ## Without baseline, log(pi(a_it, s_it)) * sum(r(s_t', a_t')) for t' = t to T

            losses.append(sum(li))   

        loss = self.hyperparameters["alpha"] * (1/n) * sum(losses)    
        #Step 5 : Backpropagation on theta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        self.replay_buffer.add_rollouts(eval_trajs)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        upd = np.mean(eval_returns)
        return upd

    def learn_actor(self, env, itr_nums, ntraj, maxtraj, imitator:ImitationAgent):
        print("Training Actor")
        for _ in range(itr_nums):
            trajs = utils.sample_n_trajectories(env, self, ntraj, maxtraj, False)
            for traj in trajs:
                state = torch.tensor(traj["observation"], dtype = torch.float, requires_grad=True).to(self.device) ## s - the current states being analysed
                action = torch.tensor(traj["action"], dtype = torch.float, requires_grad=True).to(self.device) ## action a actually taken in the trajectory
                #actions_to_replicate = torch.tensor(imitator.model.forward(state), dtype = torch.float, requires_grad=True)
                actions_to_replicate = imitator.model.forward(state)
                #print("Actions DIM : ", action.size(), " reqd action DIM : ", actions_to_replicate.size())
                actor_loss = F.mse_loss(action, actions_to_replicate, reduction="mean")
                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()
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
        self.train()
        self.sampling = True
        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        self.sampling = False
        self.hyperparameters["alpha"] = 1# np.sqrt(1/(1 + itr_num/100))
        if self.actor_critic:
            upd = self.update_ac(trajs, env)
        else:
            upd = self.update_baseline(trajs)

        print("Model's score : ",  upd)
        if  upd > self.cur_max_reward:
            print("saving Model with score : ",  upd)
            self.cur_max_reward = upd
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))

        return {'episode_loss': upd, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to


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
            
        fraction_done = (1+itr_num)/(10+itr_num) ## approaches 1 very quickly -> recalibration less and less
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
                self.rlagent.update_ac(trajs, env)


        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        self.replay_buffer.add_rollouts(eval_trajs)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        cur_reward = np.mean(eval_returns)

        print("Reward in this iteration of training = ", cur_reward)
        if  cur_reward > self.cur_max_reward:
            print("Saving model with avg score = ", cur_reward)
            self.cur_max_reward = cur_reward
            model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            torch.save(self.state_dict(), os.path.join(model_save_path, "model_"+ self.args.env_name + "_"+ self.args.exp_name+".pth"))

        trajs = utils.sample_n_trajectories(env, self, self.hyperparameters["ntraj"], self.hyperparameters["maxtraj"], False)
        return {'episode_loss': cur_reward, 'trajectories': trajs, 'current_train_envsteps': self.hyperparameters["ntraj"]} #you can return more metadata if you want to

'''
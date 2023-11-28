## RL, Policy Gradients, Imitation Learning and Bootstrapping

This Project aimed to train models for MuJoCo Environments - Hopper, Half-Cheetah and Ant - in increasing order of complexity and number of joints. 

More details about the environments and control parameters can be found at [here!](https://www.gymlibrary.dev/environments/mujoco/index.html)

The inituitive difficulty of these models is the requirement to fit a continuous action space. I take three approaches to the problem, an Imitation Learning Agent, a Policy Gradient Based Approach (both vanilla Actor Critic and Soft Actor Critic), and lastly building a better SAC model by bootstrapping it via IL.

### Setup

Dependencies are listed in the `environment_lin.yaml` file. I suggest creating a conda environment and installing these using `pip install -e .`

If unsuccesful, install by executing : 

1. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
2. `pip install -r requirements.txt`
3. `pip install -e .`

### Running Instructions 

1. The Agents are housed in `agents/`
2. The script for testing and training the model are in  `scripts/`
3. Best models are automatically saved to `best_models/`
4. `utils/` houses functionality for buffers and logging.

To run the training loops, execute `python scripts/train_agent.py --env_name <ENV> --exp_name <ALGO>  [optional tags]`
where,

- `<ENV>` is one of Hopper-v4, HalfCheetah-v4 or Ant-v4 
- `<ALGO>` is one of imitation, RL or imitation-RL
- `[optional tags]` like no_gpu and log frequencies can be used to tune runtime environment.


### Imitation Learning Agent
Key Insights and Implementational details of Imitation Agent:

- NN Model : Simple MLP , Learning Rate = 1e-3,
- Beta (DAGGER Constant) : Decreases as itr_num increases ~ (1/1+(itr_num))
- With Probability Beta, train on Expert Policy, with 1-Beta, train on your action
- After every iteration, if model performs better than previous best, checkpoint it!
- For every training iteration, randomly sample 10 trajectories from the buffer and retrain on them. 
- Best Performance : Hopper <2353>, HalfCheetah <2966>, Ant <1396>

## Reinforcement Learning Agent:

- Continuous Action space is handled by assuming that all dimensions of actions are independent of each other, 
- and using a (not too unrealistic) assumption that the actions can be assumed to be coming from a Normal Distribution
- Thus, the model outputs 2*action_dim number of values, half being the means and the other half being variances
- When querying for the next action, a sample from this distribution is returned

- Vanilla Reinforce is just too unstable
- With Baseline, doesn't seem to train on anything at all
- With Actor Critic, learns and (unstably) stays for small patches on a small reward (~1000).
- Does not explore enough to start hopping though
- Every training iteration, a random sample of 10 trajectories from the replay buffer is re-trained to combat forgetfulness

- I also incentivise higher traj lengths : by injecting reward proportional to remaining length of trajectory for that state
- The rationale behing this is that for the given (s), the model could run for (T-t) more timesteps, and if T-t is higher, the mode is able to run more

- Gradients that are added to the current policy are weighed by alpha :  sqrt(1/(1 + envsteps_so_far/1000))
- This decreases as iterations increase, inspiration from simulated-annealing.

- Implementation of Soft Actor Critic : Theoretic results from [SAC](https://arxiv.org/abs/1801.01290) suggest 
- taking minimum of multiple critics to combat over estimation, soft-transition of parameters (tau) 
- representation (reward_now * gamma * Q(s,a) - V(s)) of Advantage, and maximizing "entropy"
- Declaration : Implementation With help from [SAC](https://www.youtube.com/watch?v=ioidsRlf79o)
- Soft-Updates of parameters for stability, multi-critics to combat over-estimation

### Imitation-Seeded RL


Key Insights and Implementational Details from ImitationSeededRL:

- In the first Iteration of training, I train the Imitation attribute ("the imitator") to fit the expert policy well
- Once trained, I fit the actor model to actions suggest by the imitaton on sampled trajectories, and the critic to 
- the "rewards-to-go" obtained from these trajectories on a per-state basis.

- The ImitationSeededRL version of the Soft Actor Critic is purely my contribution

- On subsequent iterations, with probability p, perform some training iterations on the imitator and 
- "recalibrate" the actor and critic to the imitator
- With probability 1-p, let the RL component of the model explore
- As the number of iterations increase, decrease p (explore more).

Overall Optimisations:

- Greedily save the model when a test set performs better than the previous best performance

- Analysis of performance has has been meticulously performed on the trainings. Some Plots of performance vs training 
- have been attached in [https://drive.google.com/drive/folders/1PnE2SqnoBilEAzfqQvOopdSEPeQdkDtW?usp=sharing]
- This was done by modifying the train_agent.py, and plotting values from the evaluated scalar logs

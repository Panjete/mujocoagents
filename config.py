
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    'Ant-v4': {
        "imitation":{
            #You can add or change the keys here
            "hyperparameters": {
                "ntraj" : 50,
                "maxtraj" : 1000
            },
            "num_iteration": 200,

        },

        "RL":{
            #An example set of hyperparameters is given below
            #You can add or change the hyperparameters and other keys here here
            "hyperparameters": {
                "ntraj" : 50,
                "maxtraj" : 3000,
                "std_min": 0.001,
                "gamma" : 0.99,
                "alpha" : 0.0003,
                "beta" : 0.0003,
                "max_buffer_size" : 100000,
                "prob_rand_sample_training" : 0.99,
                "tau" : 0.005,
                "reward_scale" : 2
            },
            "num_iteration": 2000,
            "episode_len" : 3000
        },
        
        "imitation-RL":{
            #You can add or change the keys here
             "hyperparameters": {
                
            },
            "num_iteration": 100,

            
        }

    },


    'Hopper-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                "ntraj" : 50,
                "maxtraj" : 400
            },
            "num_iteration": 75,

        },

        "RL":{
            #You can add or change the keys here
            # ntraj is the number of trajectories sampled for 1 iterative loop
            # maxtraj is the maximum length of the trajectories that get sampled for training
            # std_min is the min variance of the gaussian distributions estimated
            # gamma is the discount factor of the trajectories
            # alpha is the gradient addition parameter
            # 1-prob_rand_sample_training is the probability of taking a random action when training
               "hyperparameters": {
                "ntraj" : 1,
                "maxtraj" : 3000,
                "std_min": 0.001,
                "gamma" : 0.99,
                "alpha" : 0.0003,
                "beta" : 0.0003,
                "max_buffer_size" : 100000,
                "prob_rand_sample_training" : 0.99,
                "tau" : 0.005,
                "reward_scale" : 2
            },
            "num_iteration": 100000,
            "episode_len" : 3000

        },

        "imitation-RL":{
            "hyperparameters": {
                "ntraj" : 1,
                "maxtraj" : 1000,
                "std_min": 0.001,
                "gamma" : 0.99,
                "alpha" : 0.0003,
                "beta" : 0.0003,
                "max_buffer_size" : 100000,
                "prob_rand_sample_training" : 0.99,
                "tau" : 0.005,
                "reward_scale" : 2
            },
            "num_iteration": 1100,
        }
    },

    'HalfCheetah-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                "ntraj" : 50,
                "maxtraj" : 400
            },
            "num_iteration": 30,
        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                "ntraj" : 50,
                "maxtraj" : 3000,
                "std_min": 0.001,
                "gamma" : 0.99,
                "alpha" : 0.0003,
                "beta" : 0.0003,
                "max_buffer_size" : 100000,
                "prob_rand_sample_training" : 0.99,
                "tau" : 0.005,
                "reward_scale" : 2
            },
            "num_iteration": 2000,
            "episode_len" : 3000

        },
        
        "imitation-RL":{
            #You can add or change the keys here
               "hyperparameters": {
                
            },
            "num_iteration": 100,

        }

    },
}
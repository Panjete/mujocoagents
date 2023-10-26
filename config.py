
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

            },
            "num_iteration": 100,

        },

        "RL":{
            #An example set of hyperparameters is given below
            #You can add or change the hyperparameters and other keys here here
             "hyperparameters": {
                'n_layers': 4,
                'hidden_size': 64,
                'learning_rate': 0.001,
                'batch_size': 3000,
                'max_ep_len': None,
                'discount' : 0.9,
            },
            "num_iteration": 200,
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
                
            },
            "num_iteration": 20,

        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                
            },
            "num_iteration": 100,

        },
    },

    'HalfCheetah-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                
            },
            "num_iteration": 100,
        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                
            },
            "num_iteration": 100,


        },
        
        "imitation-RL":{
            #You can add or change the keys here
               "hyperparameters": {
                
            },
            "num_iteration": 100,

        }

    },
}
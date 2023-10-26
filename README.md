## Dimensions

Hopper : input 11, out 3
Half Cheetah : input 17, out 6
Ant : input 27, out 8


## Imitation learning

### Hopper v4
        

* ntrajs : 50, maxlentrajs: 100 , iters : 50, lossfunction:MSE , NN: linear, beta: 0.5 reward : 742
* ntrajs : 50, maxlentrajs: 100 , iters : 50, lossfunction:MSE , NN: linear, beta: 1/(1+ sqrt(timesteps/1000)) reward : 720ish
* training iterations need to be more : steady increase phase
* training iters make reward peak at around 75. Then, rewards fall down - did go as high as 1500 when 100 timestamps

* ntrajs : 50, maxlentrajs: 100 , iters : 75, lossfunction:MSE , NN: linear, beta: 1/(1+ timesteps/1000) reward : 2200ish
*  ntrajs : 50, maxlentrajs: 400 , iters : 75, lossfunction:MSE , NN: linear, beta: 1/(1+ timesteps/1000) reward : 2400ish - more stable near 75
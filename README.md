# Using Mutual Information Minimization as Intrinsic Reward for Reinforcement Learning

![](https://github.com/Red-Pheonix/mi_minimization/blob/master/extra/exp1.gif)
![](https://github.com/Red-Pheonix/mi_minimization/blob/master/extra/exp2.gif)
![](https://github.com/Red-Pheonix/mi_minimization/blob/master/extra/exp3.gif)


This is the repository for the code of the AAI-25 PRL workshop submission "Using Mutual Information Minimization as Intrinsic Reward for Reinforcement Learning". This code is modified from [CleanRL](https://github.com/vwxyzjn/cleanrl) and uses the same prerequisites. This code will work on the `MiniGrid-SimpleCrossingS9N1-v0` environment.

# Instructions
In order to run an experiment, for example for running experiment 2 in the paper, run the following command: 


    python ppo_mi.py --seed 1 --env-id MiniGrid-SimpleCrossingS9N1-v0 --alpha 1.0 --beta -0.1


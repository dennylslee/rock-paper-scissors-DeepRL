# Introduction

The objective of this project is to construct an AI agent to play a simple 2 players rock-paper-scissors game using reinforcement learning (RL) technique - particularly the double-DQN algorithm.  

In previous design project, we have built a player using simple LSTM-based neural network that is trained using traditional supervised learning method.  The downside of that is the model is effectively static and does not adapt to model drift or fundamental behavioral changes.  With a RL-based approach, we like to see that the AI agent demostrate some ability to counter the changing strategy of the opponent and continue to generate better win rate than the opposing player. 

## Q-Learning Basics

This project utiltized the double-DQN RL algorithm as the basis for the AI agent as player 1.  DeepMind was very instrutmental in pinoeering and popularizing Q-learning with multple enhancements - thought Q-Learning was first introduced by Watkins in 1992.  See few of the popular papers arond DQN (deep Q-learning network) in the reference section.

# RPS High Level Environment 

![pic1](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/high_level_play_environment.png)


# RPS AI player architecture using Double-DQN

![pic2](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/double_dqn_architecture.png)


## Player 2 - "the opponent"


## Results



![pic3](https://github.com/dennylslee/rock-paper-scissors-DeepRL/blob/master/main_results_plot.png)

## Future Works 

## Reference

1. Mnih, et.al.  "Playing Atari with Deep Reinforcement Learning", 2013
2. Hasselt, et.al. "Deep Reinforcement Learning with Dobule Q-learning", 2015
3. Mnih, et. al. "Human-Level control through deep reinforcement learning"


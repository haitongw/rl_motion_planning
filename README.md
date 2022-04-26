# Motion Planning with Deep Reinforcement Learning
This is a course project for AER1516 Robot Motion Planning

We implemented GRU-A2C algorithm to solve the collision avoidance problem in motion planning, which can be applied to pedestrian-rich environments.

This algorithm is modified from [1]. We changed their LSTM-GA3C into GRU-A2C. The A2C implementation is modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

Advantage Actor-Critic (A2C) is a synchronous variant of A3C, which can make better use of GPU. GRU helps handle arbitrary number of nearby obstacles.

We tested this algorithm using Safety Gym[2], we made some modifications to the environment by rewriting the observations and reward functions.

### Requirements:
Python 3

[Safety Gym](https://openai.com/blog/safety-gym/)

### Reference:
[1] Everett, M., Chen, Y. F., & How, J. P. (2018, October). Motion planning among dynamic, decision-making agents with deep reinforcement learning. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 3052-3059). IEEE.

[2] [Gym API](https://www.gymlibrary.ml/)

[3] A2C Illustration:[1](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/), [2](https://openai.com/blog/baselines-acktr-a2c/)


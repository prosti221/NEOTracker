# About The Project
[Previous research](https://arxiv.org/pdf/1712.06567.pdf) on Neuroevolution show that population based search algorithms can sometimes outperform Deep Reinforcement Learning algorithms on solving some high dimensional problems. This is especially true for game playing problems, like teaching a Neural Network how to play Atari games. This suggests that certain DRL problems could have search spaces that are better suited for a population-based search approach. A problem that can very easily be formulated into a game playing problem is object tracking. The goal of this project is to see if such a phenomena can be seen for this perticular problem as well. This project therefore builds on top of a previous RL based object tracking architecture called [ADNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf), and modefies it such that it can be trained using Neuroevolution, and takes advantage of Neuroevolution's properties by extending the action decision network to be an ensamble network. The idea here is that it will use multiple diverse solutions to make predictions instead of only one, in hopes that this will increase the overall accuracy.

Previous studies in Neuroevolution have demonstrated the potential for population-based search algorithms over Deep Reinforcement Learning (DRL) algorithms in solving certain high-dimensional problems. This is particularly evident for game-playing problems, such as training a Neural Network to play Atari games. This intriguing observation suggests that specific DRL problems might have search spaces that are better suited for population-based search strategies. A problem that is easily adaptable to a game-playing framework is object tracking. 

This project aims to investigate whether these advantages observed in Neuroevolution algorithms can be harnessed for the specific challenge of object tracking. To achieve this, the project builds upon the foundations of a previous RL-based object tracking architecture known as ADNet. This Neuroevolution based tracker will first train the backbone using gradient descent based optimization. Then it freezes the backbone and trains the action-decision network using Neuroevolution. A key innovation is extending the action decision network into an ensemble network. Instead of relying on the predictions from the best fit solution, this ensemble network integrates multiple diverse solutions from the population to make predictions through majority voting.

Credit to [mare5x/adnet-rl-vot](https://github.com/mare5x/adnet-rl-vot) for the base ADNet implementation.

# Project structure
***TODO***

# Experiments
A summary of all the different model configurations that will be trained and evaluated.

| Configuration         | Description |
|------------------------|-------------|
| SL                     | SL only |
| $\text{RL}$          | RL only |
| SLRL                   | SL + RL |
| $\text{SLRL}_{fc6}$    | SL + RL applied to the last layer |
| $SLRL + \text{RL}_{fc6}$ | SL + RL, then keep backbone frozen and retrain the last layer. ($fc_6$ is randomly initialized first) |
| SLNE                   | SL + Neuroevolution; last layer is randomly initialized |
| $\text{SLNE}_{tf}$     | SL + Neuroevolution; last layer is initialized around $fc_6$ from SL training (transfer learning) |
| RLNE                   | SL + RL + Neuroevolution; Use SLRL backbone, init last layer to random |
| $\text{RLNE}_{tf}$     | SL + RL + Neuroevolution; Use SLRL backbone, init last layer around $fc_6$ from SLRL layer (transfer learning) |

- SL: Supervised Learning
- RL: Reinforcement Learning

# Results
***TODO***

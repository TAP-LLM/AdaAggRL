# AdaAggRL
source code for the paper 'Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning'
## Code Structure
```utilities.py``` contains all helper functions and defense algorithms including median, clipping median, krum and FLtrust.\
```exp_environments.py``` contains the environment used for training defense policy.\
```exp_environments_RL.py``` contains the environment used for training defense policy under RL-attack.\
```main.py``` contains code for training. \
```attack_utilities.py``` contains code for poisoning attacks.
## References
Our inverting gradients implementation is modified from https://github.com/JonasGeiping/invertinggradients

Our RL-attack implementation is modified from https://github.com/SliencerX/Learning-to-Attack-Federated-Learning

# AdaAggRL
source code for the paper '[Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34733)'
## Code Structure
```utilities.py``` contains all helper functions and defense algorithms including median, clipping median, krum and FLtrust.\
```exp_environments.py``` contains the environment used for training defense policy.\
```exp_environments_RLattack.py``` contains the environment used for training defense policy under RL-attack.\
```main.py``` contains code for training. \
```attack_utilities.py``` contains code for poisoning attacks.
## Setup Environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```
## Train
After downloading the data set and setting the parameters as required in ```main.py```, you can run the ```main.py``` file for FL training. Note: In actual run time, debug may be required to adapt to the current device and environment
## References
Our inverting gradients implementation is modified from https://github.com/JonasGeiping/invertinggradients

Our RL-attack implementation is modified from https://github.com/SliencerX/Learning-to-Attack-Federated-Learning

## Cite Format

    @inproceedings{wang2025defending,
      title={Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning},
      author={Wang, Yujing and Zhang, Hainan and Wen, Sijia and Qiu, Wangjie and Guo, Binghui},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={39},
      number={24},
      pages={25443--25451},
      year={2025}
    }

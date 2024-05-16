from exp_environments import *
from stable_baselines3 import TD3

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 128)')
    parser.add_argument('--q', type=float, default=0.1, help='i.i.d level, 0.1 means i.i.d, higher means non-i.i.d')
    parser.add_argument('--num_clients', type=int, default=100, help='number of clients(default: 100)')
    parser.add_argument('--subsample_rate', type=float, default=0.1, help='rate of clients chosen by server')
    parser.add_argument('--num_attacker', type=int, default=20)
    parser.add_argument('--num_class', type=int, default=10, help='For MNIST, Fashion-MNIST and Cifar-10')
    parser.add_argument('--fl_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset used for training (MNIST,CIFAR10...)')
    parser.add_argument('--dummy_batch_size', type=int, default=16)
    parser.add_argument('--attack', type=str, default='LMP',help='type of attack (IPM,LMP,EB)')

    args = parser.parse_args()
    return args

args = get_args()
env = FL_mnist(args)
#
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="CIFAR10-lmp-q0.1/",name_prefix='rl_model')


model = TD3("MlpPolicy", env, buffer_size=1000,
            policy_kwargs={"net_arch" : [256,128]},tensorboard_log="CIFAR10-lmp-q0.1/",
            verbose=1, gamma=0.99, action_noise = action_noise, learning_rate=1e-5, train_freq = (3, 'step'), batch_size = 64)

print('start training.......')
model.learn(total_timesteps=500, log_interval=1, callback = checkpoint_callback)
history = env.history
torch.save(torch.tensor(history['loss']), 'result/cifar10/loss_0.1_lmp_20.pth')
torch.save(torch.tensor(history['acc']), 'result/cifar10/accuracy_0.1_lmp_20.pth')

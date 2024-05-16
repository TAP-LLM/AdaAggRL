import torch
import time
import random
from collections import defaultdict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from functools import reduce
from functools import partial
from torch.autograd import Variable
import torch.optim as optim
import math
from typing import Optional, Tuple, List
from ray.util.multiprocessing import Pool
import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


# 模型设置参数
DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')


def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':  # DEFAULT_CONFIG中scoring_choice='loss'
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        # self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):  # DummySet,
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape)  # , DummySet)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))

                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            all_labels = []
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                all_labels.append(labels)
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            #print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            if torch.numel(scores) == 0:
                stats['opt'] = 1000
                print("badbadbadbad")
                return x[-1].detach(), stats, all_labels[-1]
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]
            label_optimal = all_labels[optimal_index]

        #print(f'Total time: {time.time() - start_time}.')
        return x_optimal.detach(), stats, label_optimal

    def _init_images(self, img_shape):  # , DummySet):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        # elif self.config['init'] == 'pre':
        #     # print(torch.stack([DummySet for _ in range(self.config['restarts'])]).shape)
        #     # return torch.unsqueeze(DummySet,0)
        #     return torch.stack([DummySet for _ in range(self.config['restarts'])])
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            # labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)
            labels = torch.randn((self.num_images, output_test.shape[1])).to(**self.setup).requires_grad_(True)
            # print(labels.shape)
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=self.config['lr'], momentum=0, nesterov=False)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels], lr=self.config['lr'])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=self.config['lr'], momentum=0, nesterov=False)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial], lr=self.config['lr'])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142],
                                                             gamma=0.1)  # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    # if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                    #     print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')
                    # elif iteration % 10 == 0 and self.config['optim'] == 'LBFGS':
                    #     print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            # print(label.shape)
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * total_variation(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return total_variation(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        # print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    offset = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                if input_gradient[i].shape != trial_gradient[i - offset].shape:
                    offset += 1
                    continue
                else:
                    costs += ((trial_gradient[i - offset] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                if input_gradient[i].shape != trial_gradient[i - offset].shape:
                    offset += 1
                    continue
                else:
                    costs -= (trial_gradient[i - offset] * input_gradient[i]).sum() * weights[i]
                    pnorm[0] += trial_gradient[i - offset].pow(2).sum() * weights[i]
                    pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def _build_groups_by_q(trainset, q, num_class=10):
    groups = []
    for _ in range(num_class):
        groups.append([])
    for img, lable in trainset:
        if random.random() < (q - 0.1) * num_class / (num_class - 1):
            groups[lable].append((img, lable))
        else:
            groups[random.randint(0, num_class - 1)].append((img, lable))
    return groups


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNISTClassifier_small(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 3)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MNISTClassifier(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_real(net, trainloader, epochs, lr):
    """Train the network on the training set."""
    # criterion = torch.nn.CrossEntropyLoss()
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    num_pic = 0
    steps = 0
    labels = None
    for _ in range(epochs):
        for images, labels in trainloader:
            steps += 1
            #images, labels = next(iter(trainloader))
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            num_pic += len(images)
            # labels_in = _label_to_onehot(labels, 10).to(DEVICE)
            optimizer.zero_grad()
            # input = images.expand(-1,3,-1,-1)
            loss = criterion(net(images), labels)
            # g = torch.autograd.grad(loss, net.parameters(), retain_graph = True) #新加的
            # dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph = True)
            loss.backward()
            optimizer.step()

    return num_pic, labels, steps  # 新加的


def train(net, train_iter, epochs, lr, mode = True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        #for images, labels in trainloader:
        try:
            images, labels = next(train_iter)
        except:
            train_iter.seek(0)
            images, labels = next(train_iter)
        #print(labels)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if mode:
            loss = criterion(net(images), labels)
        else:
            loss = -criterion(net(images), labels)
        loss.backward()
        #clipping
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

def train_real_ga(net, trainloader, epochs, lr):
    """Train the network on the training set."""
    net.train()
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        #for images, labels in trainloader:
        images, labels = next(iter(trainloader))
        #print(labels)
        # print('use for train')
        # plt.show(tt(images[0].cpu()))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #for i in range(len(images)):
              #print(dummy_labelset[i])
             # plt.subplot(1, len(images), i + 1)
             # plt.imshow(images[i].reshape(28,28).cpu().detach().numpy())
              #plt.axis('off')
              #plt.savefig("true.png")
        #labels_in = _label_to_onehot(labels, 10).to(DEVICE)
        optimizer.zero_grad()
        loss = -criterion(net(images), labels)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), retain_graph = True)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph = True)
        loss.backward()
        optimizer.step()


def test(net, valloader):
    """Validate the network on the 10% training set."""
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
            # data=next(iter(valloader))
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # print(len(images))
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #average_loss = loss / total
        accuracy = correct / total
    return loss, accuracy


def get_parameters(net):
    # for _, val in net.state_dict().items():
    # if np.isnan(val.cpu().numpy()).any(): print(val)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def compare_weights(weight1, weight2):
    r = []
    for i in range(len(weight1)):
        r.append((weight1[i] == weight2[i]).all())
    return r
# def get_trainable_parameters(net):
#     # for _, val in net.state_dict().items():
#     # if np.isnan(val.cpu().numpy()).any(): print(val)
#     return [val.cpu() for _, val in enumerate(net.parameters())]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    # print(net.state_dict().keys())
    for i in range(len(parameters)):
        if len(parameters[i].shape) == 0:
            parameters[i] = np.asarray([parameters[i]])
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # print((state_dict))
    net.load_state_dict(state_dict, strict=True)


def average(new_weights):
    fractions = [1 / len(new_weights) for _ in range(len(new_weights))]
    fraction_total = np.sum(fractions)

    # Create a list of weights, each multiplied by the related fraction
    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights

def weighted_weights(new_weights, fractions):
    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]
    return weighted_weights
def aggeregate(new_weights, fractions):
    fraction_total = np.sum(fractions)

    # Create a list of weights, each multiplied by the related fraction
    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights


def common(a, b):
    c = [value for value in a if value in b]
    return c


def weights_to_vector(weights):
    """Convert NumPy weights to 1-D Numpy array."""
    Lis=[np.ndarray.flatten(ndarray) for ndarray in weights]
    return np.concatenate(Lis, axis=0)
def vector_to_weights(vector,weights):
    """Convert 1-D Numpy array tp NumPy weights."""
    indies = np.cumsum([0]+[layer.size for layer in weights]) #indies for each layer of a weight
    Lis=[np.asarray(vector[indies[i]:indies[i+1]]).reshape(weights[i].shape) for i in range(len(weights))]
    return Lis


def exclude(a,b):
    c = [value for value in a if value not in b]
    return c

def check_attack(cids, att_ids):
    return  np.array([(id in att_ids) for id in cids]).any()


def Krum(old_weight, new_weights, num_round_attacker):
    """Compute Krum average."""

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    scrs=[]
    for i in grads:
        scr=[]
        for j in grads:
            dif=weights_to_vector(i)-weights_to_vector(j)
            sco=np.linalg.norm(dif)
            scr.append(sco)
        top_k = sorted(scr)[1:len(grads)-2-num_round_attacker]
        scrs.append(sum(top_k))
    chosen_grads= grads[scrs.index(min(scrs))]
    krum_weights = [w1-w2 for w1,w2 in zip(old_weight, chosen_grads)]
    return krum_weights


def Median(old_weight, new_weights):
    """Compute Median average."""

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))
    Median_weights = [w1-w2 for w1,w2 in zip(old_weight, med_grad)]
    return Median_weights


def Clipping_Median(old_weights, new_weights):
    max_norm=2
    grads=[]
    for new_weight in new_weights:
        #print(len(new_weight))
        #print(len(old_weights))
        norm_diff=np.linalg.norm(weights_to_vector(old_weights)-weights_to_vector(new_weight))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)


    # fractions=[1/int(num_clients*subsample_rate) for _ in range(int(num_clients*subsample_rate))]
    # fraction_total=np.sum(fractions)
    #
    # # Create a list of weights, each multiplied by the related fraction
    # weighted_grads = [
    #     [layer * fraction for layer in grad] for grad, fraction in zip(grads, fractions)
    # ]
    #
    # # Compute average weights of each layer
    # aggregate_grad = [
    #     reduce(np.add, layer_updates) / fraction_total
    #     for layer_updates in zip(*weighted_grads)
    # ]

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, med_grad)]


    return Centered_weights



def Clipping(old_weights, new_weights,cids):
    max_norm=2
    grads=[]
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)


    fractions=[1/len(cids) for _ in range(len(cids))]
    fraction_total=np.sum(fractions)

    # Create a list of weights, each multiplied by the related fraction
    weighted_grads = [
        [layer * fraction for layer in grad] for grad, fraction in zip(grads, fractions)
    ]

    # Compute average weights of each layer
    aggregate_grad = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_grads)
    ]

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, aggregate_grad)]


    return Centered_weights


def FLtrust(net, old_weight, new_weights, valid_loader, g_weight = None, lr = 0.05):

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    if g_weight == None:
        #net = torch.load("small_mnist_init").to(**setup)
        #net = torch.load("emnist_init").to(**setup)
        set_parameters(net, old_weight)
        #valid_loader.seek(0)
        train(net, valid_loader, epochs=1, lr=lr)
        new_weight = get_parameters(net)
    else:
        new_weight=g_weight
    #print("2")
    #print(new_weight[-1])
    server_grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]

    vec_grads = [weights_to_vector(grad) for grad in grads]
    vec_server_grad = weights_to_vector(server_grad)

    #TS = [cos_sim(vec_grad ,vec_server_grad) for vec_grad in vec_grads]
    #print(TS)
    TS = [relu(cos_sim(vec_grad, vec_server_grad)) for vec_grad in vec_grads]

    normlized_vec_grads = [np.linalg.norm(vec_server_grad) / (np.linalg.norm(vec_grad)+1e-10) * vec_grad for vec_grad in
                           vec_grads]

    # normlized_vec_grads = []
    # for vec_grad in vec_grads:
    #     if np.any(np.linalg.norm(vec_grad)*vec_grad == 0):
    #         normlized_vec_grads.append(np.linalg.norm(vec_server_grad)/(np.linalg.norm(vec_grad)*vec_grad + 1e-5))
    #     else:
    #         normlized_vec_grads.append(np.linalg.norm(vec_server_grad) / (np.linalg.norm(vec_grad) * vec_grad))
    normlized_grads = [vector_to_weights(vec_grad, server_grad) for vec_grad in normlized_vec_grads]
    #client_weights=[np.linalg.norm(vec_server_grad)/np.linalg.norm(vec_grad)*TC for vec_grad, TC in zip(vec_grads, TS)]

    #print(client_weights)

    TS_total=np.sum(TS)
    #print(TS)
    #if TS_total<0.5: TS_total=0

    # Create a list of weights, each multiplied by the related fraction
    weighted_grads = [
        [layer * TC for layer in grad] for grad, TC in zip(normlized_grads, TS)
    ]

    # Compute average weights of each layer
    FLtrust_grad = [
        reduce(np.add, layer_updates) / max(TS_total, 1e-8)
        for layer_updates in zip(*weighted_grads)
    ]

    FLtrust_weights = [w1-w2 for w1,w2 in zip(old_weight, FLtrust_grad)]



    return FLtrust_weights


def relu(x): return max(0.0, x)


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity."""

    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a)+ 1e-10 #|x|
    norm_b = np.linalg.norm(b)+ 1e-10 #|y|
    # if norm_a * norm_b == 0:
    #     return dot_product / (norm_a * norm_b + 1e-5)
    return dot_product / (norm_a * norm_b)

# 用于计算MMD的函数
def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: torch.sum(torch.square(x), dim=1)

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    # x will be num_samples x num_features x 1,
    # and y will be 1 x num_features x num_samples (after broadcasting).
    # After the substraction we will get a
    # num_x_samples x num_features x num_y_samples matrix.
    # The resulting dist will be of shape num_y_samples x num_x_samples.
    # and thus we need to transpose it again.
    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)  # torch.unsqueeze(torch.tensor([sigma], dtype=torch.float32), 1)

    dist = compute_pairwise_distances(x, y).float()
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))

    return torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape)


def mmd_origin(x, y, kernel=gaussian_kernel_matrix):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    #sigmas = torch.FloatTensor([1, 5, 10, 15, 20])
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def maximum_mean_discrepancy(source, target):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=Variable(torch.FloatTensor(sigmas))
    )
    cost = mmd_origin(source, target, kernel=gaussian_kernel)
    # We do not allow the loss to become negative.
    if cost < 0:
        cost = 0
    cost = 2 * torch.cos(torch.tanh(0.5*cost)) - 1
    return cost


def train_net_defender(net, global_model, train_dataloader, epochs, lr, args_optimizer, args):
    net.cuda()
    global_model.cuda()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().cuda()
    cos = torch.nn.CosineSimilarity(dim=-1).cuda()
    kl_criterion = nn.KLDivLoss(reduction="batchmean").cuda()

    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            target = target.long()
            optimizer.zero_grad()
            outputs, SD_outputs, feats = net(x, get_feat=True, SD=True)
            SD_p_output = F.softmax(SD_outputs / args.temperature, dim=1)
            SD_logp = F.log_softmax(SD_outputs / args.temperature, dim=1)
            p_output = F.softmax(outputs / args.temperature, dim=1)
            logp_output = F.log_softmax(outputs / args.temperature, dim=1)

            with torch.no_grad():
                logp_global = global_model(x)
                logp_global = F.softmax(logp_global / args.temperature, dim=1)
                logp_global = logp_global.detach()

            alpha = cos(logp_global, F.one_hot(target, num_classes=10)).unsqueeze(1)
            targer_g = (1 - alpha) * F.one_hot(target, num_classes=10) + alpha * logp_global
            loss_gkd = -torch.mean(torch.sum(SD_logp * targer_g, dim=1))
            loss = criterion(outputs, target) + loss_gkd + kl_criterion(logp_output, SD_p_output.detach())

            loss.backward(retain_graph=True)
            targets_fast = target.clone()
            randidx = torch.randperm(target.size(0))
            for n in range(int(target.size(0) * 0.5)):
                num_neighbor = 10
                idx = randidx[n]
                feat = feats[idx]
                feat.view(1, feat.size(0))
                feat.data = feat.data.expand(target.size(0), feat.size(0))
                dist = torch.sum((feat - feats) ** 2, dim=1)
                _, neighbor = torch.topk(dist.data, num_neighbor + 1, largest=False)
                targets_fast[idx] = target[neighbor[random.randint(1, num_neighbor)]]

            fast_loss = criterion(outputs, targets_fast)
            grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True,
                                        only_inputs=True, allow_unused=True)

            for grad in grads:
                if grad == None:
                    continue
                grad = grad.detach()
                grad.requires_grad = False

            fast_weights = OrderedDict(
                (name, param - args.lr * grad) for ((name, param), grad) in zip(net.named_parameters(), grads) if
                grad != None)
            fast_out, SD_fast_out = net(x, fast_weights, SD=True)

            logp_fast = F.log_softmax(fast_out, dim=1)
            meta_loss = criterion(fast_out, target)
            meta_loss.backward()

            optimizer.step()

    # train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, _ = compute_accuracy(net, test_dataloader, device=device)
    # net.to('cpu')
    #
    # return train_acc, test_acc


def multiprocess_evaluate(model, weights, x_test, y_test):
    set_parameters(model,weights)
    preds = model(x_test).cpu().detach().numpy()
    spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
    for i in range(len(preds)):
        pred = np.argmax(preds[i])
        true = np.argmax(y_test[i])
        spec_label_all_count[true] = spec_label_all_count[true] + 1
        spec_label_loss_count[true] += -(math.log(max(preds[i][true], 0.0001)))
        if true == pred:
            spec_label_correct_count[true] = spec_label_correct_count[true] + 1
    spec_label_accuracy = []
    spec_label_loss = []
    all_sum = 0
    all_acc_correct = 0
    all_loss_correct = 0
    for i in range(len(spec_label_all_count)):
        all_sum += spec_label_all_count[i]
        spec_label_accuracy.append(spec_label_correct_count[i] / spec_label_all_count[i])
        all_acc_correct += spec_label_correct_count[i]
        spec_label_loss.append(spec_label_loss_count[i] / spec_label_all_count[i])
        all_loss_correct += spec_label_loss_count[i]
    print(f"acc client: {all_acc_correct / all_sum}")
    print(f"spec label client {spec_label_accuracy}")
    # np.mean(spec_label_loss) if we want each label to mean as much, use
    # all_loss_correct/all_sum instead as first return if you want to promote the distribution of test data
    return np.mean(spec_label_loss), {"accuracy": all_acc_correct / all_sum}, spec_label_accuracy, spec_label_loss


class Poison_detect:
    # s1_factor determines by how much more we want to favor the stronger client updates
    # s2 determines how important it is for labels to not fall behind
    def __init__(self, x_val, y_val, model, s1_overall=2, s1_label=3, s2=3):
        self.model = model
        self.evclient = Poison_detect.get_eval_fn(self.model, x_val, y_val)
        self.x_test = x_val
        self.y_test = y_val
        self.no_labels = int((y_val[0].size())[0])#len(y_val[0])
        self.s1_overall = s1_overall
        self.s1_label = s1_label
        self.s2 = s2
        self.pre_reset_s2 = s2

    """
    Input is results: List[Tuple]. The list contains Tuples where first element in the tuple is client ID and second
    element in tuple is a list of ndarrays for the updated client model of client with said ID. last_agg_w is the global
    models weights for the last round, used to calculate norms. Returns an aggregation of each updated client model based
    on input parameters
    """

    def calculate_new_aggregated(self, results: List[Tuple], last_agg_w: list):
        label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss = self.calculate_accs(results)
        adaptives2Loss = []
        adaptives2Parts = []
        weights = []
        # this could be parallelized
        adaptives2Tests = [self.s2, max(1, self.s2 - 0.5), self.s2 + 0.5, 3, self.pre_reset_s2]
        i = 0
        for elem in adaptives2Tests:
            self.s2 = elem
            points = {}
            points, overall_mean = self.get_points_overall(loss_dict, results, points=points)
            points = self.get_points_label(label_loss_dict, results, overall_mean, points)
            part_agg = self.points_to_parts(points)
            agg_copy_weights = self.agg_copy_weights(results, part_agg, last_agg_w)
            weights.append(agg_copy_weights)
            loss, acc, _, _ = self.evclient(agg_copy_weights)
            adaptives2Parts.append(part_agg)
            adaptives2Loss.append(loss)
            print(f"acc on {elem}: {acc}")
            i = i + 1
        idx_max = np.argmin(adaptives2Loss)
        if idx_max == 3:
            self.pre_reset_s2 = self.s2
        self.s2 = adaptives2Tests[idx_max]
        print(f"self.s2 is now: {self.s2}")
        return weights[idx_max]

    def agg_copy_weights(self, results, part_agg, last_weights):
        _, norms_dict = self.calculate_avg_norms1(results, last_weights)
        ret_weights = []
        for elem in norms_dict:
            for i in range(len(norms_dict[elem])):
                if i < len(ret_weights):
                    ret_weights[i] = np.add(ret_weights[i], norms_dict[elem][i] * part_agg[elem])
                else:
                    ret_weights.append(norms_dict[elem][i] * part_agg[elem])
        for i in range(len(ret_weights)):
            ret_weights[i] = np.add(ret_weights[i], last_weights[i])
        return ret_weights

    def get_norms(self, weights, last_weights):
        norms = []
        for i in range(len(weights)):
            norms.append(np.subtract(weights[i], last_weights[i]))
        return norms

    def calculate_avg_norms1(self, results, last_weights):
        norms_dict = {}
        norms_list = []
        for elem in results:
            norm = self.get_norms(elem[1], last_weights)
            norms_dict[elem[0]] = norm
            norms_list.append(norm)
        norms_avg = copy.deepcopy(norms_list[0])
        for w_indx in range(len(norms_list[0])):
            for c_indx in range(1, len(norms_list)):
                norms_avg[w_indx] = np.add(norms_avg[w_indx], norms_list[c_indx][w_indx])
        for i in range(len(norms_avg)):
            norms_avg[i] = norms_avg[i] / len(norms_list)
        return norms_avg, norms_dict

    def points_to_parts(self, points):
        part_agg = {}
        # make sure no client has negative points
        for elem in points:
            points[elem] = max(0, points[elem])
        sum_points = 0
        for elem in points:
            sum_points += points[elem]
        sum_points = max(000.1, sum_points)
        for elem in points:
            part_agg[elem] = (points[elem] / sum_points)
        return part_agg

    def get_points_overall(self, nodes_acc, results, points={}):
        # overall points
        # calculate mean absolute deviation for middle 80% of clients
        mean_calc = []
        for elem in nodes_acc:
            mean_calc.append(nodes_acc[elem])
        mean = np.mean(mean_calc)
        all_for_score = []
        for elem in mean_calc:
            # if loss then (mean - elem), if accuracy (mean - elem)
            all_for_score.append(mean - elem)
        mad_calc = all_for_score.copy()
        for i in range(len(mad_calc)):
            mad_calc[i] = abs(mad_calc[i])
        no_elems = round(len(mad_calc))
        mad_calc.sort()
        mad_calc = mad_calc[:no_elems]
        mad = np.mean(mad_calc)
        slope = self.s1_overall / mad
        for i in range(len(all_for_score)):
            points[results[i][0]] = points.get(results[i][0], 0) + slope * all_for_score[i] + 10
        # individual label points
        return points, mean

    def get_points_label(self, label_acc_dict, results, overall_mean, points):
        # individual label points
        for i in range(self.no_labels):
            mean_calc = []
            for elem in label_acc_dict:
                mean_calc.append(label_acc_dict.get(elem)[i])
            mean = np.mean(mean_calc)
            all_for_score = []
            for elem in mean_calc:
                all_for_score.append(mean - elem)
            mad_calc = all_for_score.copy()
            for j in range(len(mad_calc)):
                mad_calc[j] = abs(mad_calc[j])
            no_elems = round(len(mad_calc))
            mad_calc.sort()
            mad_calc = mad_calc[:no_elems]
            mad = np.mean(mad_calc)
            slope = self.s1_label / mad

            dif = (mean - overall_mean)
            x = ((overall_mean + dif) / overall_mean)
            factor = x ** self.s2
            for k in range(len(all_for_score)):
                points[results[k][0]] = points.get(results[k][0], 0) + (max(1, factor)) * slope * all_for_score[k] + 10
        return points

    def par_results_ev(self, result):
        loss, acc, lab_acc, lab_loss = multiprocess_evaluate(self.model, result[1], self.x_test, self.y_test)
        return [result[0], loss, acc, lab_acc, lab_loss]

    def calculate_accs(self, results):
        label_acc_dict = {}
        nodes_acc = {}
        loss_dict = {}
        label_loss_dict = {}
        #pool = Pool(ray_address="auto")
        evaluated = []
        for result in results:
            evaluated.append(self.par_results_ev(result))
        for elem in evaluated:
            label_acc_dict[elem[0]] = elem[3]
            nodes_acc[elem[0]] = elem[2].get('accuracy')
            loss_dict[elem[0]] = elem[1]
            label_loss_dict[elem[0]] = elem[4]
        # redundant:)
        last_loss = 0
        last_label_loss = 0
        return label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss

    @staticmethod
    def get_eval_fn(model, x_test, y_test):
        """Return an evaluation function for server-side evaluation."""

        def evaluate(weights) -> Optional[Tuple[float, float]]:
            set_parameters(model,weights)
            preds = model(x_test).cpu().detach().numpy()
            spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
            for i in range(len(preds)):
                pred = np.argmax(preds[i])
                true = np.argmax(y_test[i])
                spec_label_all_count[true] = spec_label_all_count[true] + 1
                spec_label_loss_count[true] += -(
                    math.log(max(preds[i][true], 0.0001)))  # 0.0001 to avoid divide by zero
                if true == pred:
                    spec_label_correct_count[true] = spec_label_correct_count[true] + 1
            spec_label_accuracy = []
            spec_label_loss = []
            all_sum = 0
            all_acc_correct = 0
            all_loss_correct = 0
            for i in range(len(spec_label_all_count)):
                all_sum += spec_label_all_count[i]
                spec_label_accuracy.append(spec_label_correct_count[i] / spec_label_all_count[i])
                all_acc_correct += spec_label_correct_count[i]
                spec_label_loss.append(spec_label_loss_count[i] / spec_label_all_count[i])
                all_loss_correct += spec_label_loss_count[i]
            # np.mean(spec_label_loss) if we want each label to mean as much, use
            # all_loss_correct/all_sum instead as first return if you want to promote the distribution of test data
            return np.mean(spec_label_loss), {
                "accuracy": all_acc_correct / all_sum}, spec_label_accuracy, spec_label_loss

        return evaluate
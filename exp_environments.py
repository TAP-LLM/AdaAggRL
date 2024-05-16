import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torch import nn
from utilities import *
import math
from resnetcifar import *
from attack_utilities import *
import gym
from gym import spaces
from gym.utils import seeding
from inversefed.data.data_processing import construct_dataloaders
from utilities import _build_groups_by_q
from torch.utils.tensorboard import SummaryWriter
import time
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup = dict(device=DEVICE, dtype=torch.float)

class FL_mnist(gym.Env):

    def __init__(self, args):  # att_trainset, validset):

        self.rnd = 0
        self.args = args
        # self.weights_dimension = 1290  # 510, 1290, 21840

        high = 0.95
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(5,),
            dtype=np.float32
        )

        high = 1
        low = 0
        self.observation_space = spaces.Box(low=low, high=high, shape=(int(args.num_clients * args.subsample_rate), 4))# self.seed()
        # ***********************************************************************************
        random.seed(150)
        att_ids = random.sample(range(args.num_clients), args.num_attacker)
        self.att_ids = list(np.sort(att_ids, axis=None))
        print('attacker ids: ', self.att_ids)
        self.trainset, self.testset = construct_dataloaders(args.dataset, data_path='./data')
        # 数据集处理
        cc = torch.cat([self.trainset[i][0].reshape(-1) for i in range(len(self.trainset))], dim=0)
        dm = (torch.mean(cc, dim=0).item(),)
        ds = (torch.std(cc, dim=0).item(),)

        train_groups = _build_groups_by_q(self.trainset, args.q, num_class=args.num_class)
        test_groups = _build_groups_by_q(self.testset, args.q,num_class=args.num_class)
        trainloaders, testloaders = [], []
        num_group_clients = int(args.num_clients / args.num_class)
        for gid in range(args.num_class):
            num_traindata = int(len(train_groups[gid]) / num_group_clients)
            num_testdata = int(len(test_groups[gid]) / num_group_clients)
            for cid in range(num_group_clients):
                ids = list(range(cid * num_traindata, (cid + 1) * num_traindata))
                client_trainset = torch.utils.data.Subset(train_groups[gid], ids)
                ids = list(range(cid * num_testdata, (cid + 1) * num_testdata))
                client_testset = torch.utils.data.Subset(test_groups[gid], ids)
                trainloaders.append(
                    torch.utils.data.DataLoader(client_trainset, batch_size=args.batch_size, shuffle=True,
                                                drop_last=True))
                testloaders.append(
                    torch.utils.data.DataLoader(client_testset, batch_size=args.dummy_batch_size, shuffle=True,
                                                drop_last=True))

        self.trainloaders = trainloaders
        self.testloaders = testloaders
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=args.batch_size, shuffle=False,
                                                      drop_last=True)

        self.config = dict(signed=True,
                           boxed=True,
                           cost_fn='sim',
                           indices='def',
                           weights='equal',
                           lr=0.05,
                           optim='adam',
                           restarts=1,
                           max_iterations=30,
                           total_variation=1e-6,
                           init='zeros',
                           filter='none',
                           lr_decay=True,
                           scoring_choice='loss')

        self.dm = torch.as_tensor(dm, **setup)[:, None, None]
        self.ds = torch.as_tensor(ds, **setup)[:, None, None]
        self.image_shape = tuple(client_trainset[0][0].shape)

        if args.dataset == 'CIFAR10':
            extract_feature = torchvision.models.resnet18(pretrained=True)
        elif args.dataset == 'EMNIST':
            extract_feature = MNISTClassifier()
            state_dict = torch.load('extract_feature_emnist.pt')
            extract_feature.load_state_dict(state_dict)
        else:
            extract_feature = MNISTClassifier()
            state_dict = torch.load('extract_feature.pt')
            extract_feature.load_state_dict(state_dict)
        self.extract_feature = nn.Sequential(*list(extract_feature.children())[:-1])
        self.extract_feature.to(DEVICE)
        self.extract_feature.eval()
        self.tensorboard = SummaryWriter("ImageNet_IPM_loss_acc/")
        self.history = {'loss':[],'acc':[]}
        # ***********************************************************************************

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        args = self.args
        self.rnd += 1
        # reward
        weights_lis = []
        print(self.cids)
        for cid in self.cids:
            weights_lis.append(self.weights_dict[cid])
        action_0 = torch.tensor(action[:4])
        action_0 = torch.softmax(action_0, dim=0).numpy()
        k = np.dot(self.old_state, action_0)
        # print(k)
        k = torch.tensor(k)
        k = (k-torch.min(k))/(torch.max(k)-torch.min(k))
        k = 0.5 * (1 - torch.cos(3.14 * k))
        k = k / torch.sum(k)
        delta = torch.max(k) * torch.tensor(action)[4]
        for i in range(len(self.cids)):
            if k[i] <= delta:
                k[i] = 0
                self.client_state[self.cids[i]]['flag'] += 1
            else:
                k[i] = k[i] * (0.9 ** self.client_state[self.cids[i]]['flag'])
                self.client_state[self.cids[i]]['flag'] = max(0, self.client_state[self.cids[i]]['flag']-1)
        print(k)
        new_weight = aggeregate(weights_lis, k.numpy().tolist())
        self.aggregate_weights = copy.deepcopy(new_weight)
        set_parameters(self.net, self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.testloader)
        reward = self.loss - new_loss
        print('self.loss = {}, new_loss = {}, reward = {}'.format(self.loss, new_loss, reward))
        self.loss = copy.deepcopy(new_loss)
        self.history['loss'].append(new_loss)
        self.history['acc'].append(new_acc)
        self.tensorboard.add_scalar("loss", self.loss, self.rnd)
        self.tensorboard.add_scalar("accuracy", new_acc, self.rnd)
        print('===================================================================================================')
        print('rnd = {}, loss = {}, accuracy = {}'.format(self.rnd, new_loss, new_acc))
        # **************************************************************************************************************
        # Clients' Operation
        old_weights = copy.deepcopy(self.aggregate_weights)

        self.cids = random.sample(range(args.num_clients), int(args.num_clients * args.subsample_rate))
        start = time.perf_counter()
        weights_dict = {}
        steps = 1
        non_att = exclude(self.cids, self.att_ids)
        for cid in non_att:  # if there is no attack
            set_parameters(self.net, old_weights)
            num_pic, labels, steps = train_real(self.net, self.trainloaders[cid], epochs=1, lr=args.lr)
            new_weight = get_parameters(self.net)
            # weights_lis.append(new_weight)
            weights_dict[cid] = copy.deepcopy(new_weight)

        if check_attack(self.cids, self.att_ids):
            real_att = []
            for cid in common(self.cids,self.att_ids):
                if self.client_state[cid]['times'] >= 1:
                    real_att.append(cid)
                else:
                    non_att.append(cid)
                    set_parameters(self.net, old_weights)
                    num_pic, labels, steps = train_real(self.net, self.trainloaders[cid], epochs=1, lr=args.lr)
                    new_weight = get_parameters(self.net)
                    # weights_lis.append(new_weight)
                    weights_dict[cid] = copy.deepcopy(new_weight)
            print(real_att)
            if len(real_att)>0:
                set_parameters(self.net, old_weights)
                if args.attack == 'IPM':
                    attak_dict = IPM_attack(self.net, old_weights, self.cids, real_att)
                elif args.attack == 'LMP':
                    attak_dict = LMP_attack(self.net, old_weights, self.cids, real_att, self.trainloaders,
                                            list(weights_dict.values()))
                elif args.attack == 'EB':
                    attak_dict = EB_attack(self.net, old_weights, self.cids, real_att, self.trainloaders,
                                           self.testloaders)
                # weights_lis = weights_lis + list(attak_dict.values())
                weights_dict = {**weights_dict, **attak_dict}
        else:
            for cid in self.cids:
                set_parameters(self.net, old_weights)
                num_pic, labels, steps = train_real(self.net, self.trainloaders[cid], epochs=1, lr=args.lr)
                new_weight = get_parameters(self.net)
                # weights_lis.append(new_weight)
                weights_dict[cid] = copy.deepcopy(new_weight)

        # Server's Operation
        score_cid = {}
        global_feature = 0
        for cid in self.cids:
            # server computes gradient
            weight_cid = weights_dict[cid]
            input_gradient = [torch.from_numpy((w2 - w1) / (args.lr * steps)).to(**setup) for w1, w2 in
                              zip(weight_cid, old_weights)]  # 做过修改
            input_gradient = [grad.detach() for grad in input_gradient]

            # server learns the distribution
            set_parameters(self.net, weight_cid)
            self.net.eval()
            self.net.zero_grad()
            #print('recovering {} client distribution'.format(cid))
            rec_machine = GradientReconstructor(self.net, (self.dm, self.ds), self.config,
                                                num_images=args.dummy_batch_size)  # args.dummy_batch_size)
            output, stats, recovered_labels = rec_machine.reconstruct(input_gradient, None,
                                                                      img_shape=self.image_shape)  # dummy_batch,

            # Image features are extracted to calculate MMD
            feature = self.extract_feature(output).view(args.dummy_batch_size, -1).detach()
            self.client_state[cid]['current_feature'] = (copy.deepcopy(feature),stats['opt'])
            global_feature += feature
        global_feature = global_feature / len(self.cids)

        for cid in self.cids:
            self.client_state[cid]['global_feature'] = copy.deepcopy(global_feature)

            # The server calculates the distribution difference between the two recovery datasets before and after the client
            if self.client_state[cid]['times'] == 0:
                sim_lc = 0.9
                sim_lg = 0.9
                sim_cg = maximum_mean_discrepancy(self.client_state[cid]['current_feature'][0].cpu(), global_feature.cpu())

                score_cid[cid] = [1-self.client_state[cid]['current_feature'][1],sim_lc, sim_lg, sim_cg]
            else:
                sim_lc = maximum_mean_discrepancy(self.client_state[cid]['local_feature'][0].cpu(),
                                                  self.client_state[cid]['current_feature'][0].cpu())
                sim_lg = maximum_mean_discrepancy(self.client_state[cid]['local_feature'][0].cpu(), global_feature.cpu())
                sim_cg = maximum_mean_discrepancy(self.client_state[cid]['current_feature'][0].cpu(), global_feature.cpu())
                print('similarity of client {} is lc={} lg={} cg={}'.format(cid, sim_lc, sim_lg, sim_cg))

                t = torch.tensor(self.client_state[cid]['times'])
                ft = -2 * torch.cos(torch.tanh(t)) + 2
                ft = ft.numpy()
                if sim_lc >= 0.9:
                    score_cid[cid] = [0, 0, sim_lg, sim_cg]
                else:
                    score_cid[cid] = [1-self.client_state[cid]['current_feature'][1],sim_lc, sim_lg, sim_cg]
            self.client_state[cid]['local_feature'] = copy.deepcopy(self.client_state[cid]['current_feature'])
            self.client_state[cid]['times'] += 1
        end = time.perf_counter()
        print(str(end-start))
        new_state = []
        for cid in self.cids:
            value = score_cid[cid]
            new_state.append(value)
        new_state = np.array(new_state)

        done = False
        if self.rnd >= 1000:
            done = True  # 15, 25, 75
            _, acc = test(self.net, self.testloader)
            print(self.rnd, new_loss, acc)

        self.old_state = new_state
        self.weights_dict = weights_dict
        if reward < -10:
            done = False
            print('action = {}'.format(action))

        if reward < -1000:
            done = True
            print('action = {}'.format(action))

            return self.old_state, reward, done, {}


        return new_state, reward, done, {}

        # ****************************************************************************************************************

    def reset(self):
        args = self.args
        self.rnd = 0
        #random.seed(150)
        if args.dataset == 'CIFAR10':
            model = torchvision.models.resnet18(num_classes=10)#, pretrained=True)
            # model = ResNet18_cifar(class_num=10)
            # model.conv1 = nn.Conv2d(model.conv1.in_channels,model.conv1.out_channels,3,1,1)
            # model.maxpool = nn.Identity()
            # model.fc = nn.Linear(model.fc.in_features,10)
            self.net = model
        else:
            self.net = MNISTClassifier()
        self.net = self.net.to(DEVICE)
        self.aggregate_weights = get_parameters(self.net)

        self.cids = random.sample(range(args.num_clients), int(args.num_clients * args.subsample_rate))
        self.client_state = [{'flag': 0, 'times': 0, 'local_feature': torch.tensor(0.), 'current_feature': torch.tensor(0.),
                              'global_feature': torch.tensor(0.)} for i in
                             range(args.num_clients)]
        self.loss, self.acc = test(self.net, self.testloader)
        self.history['loss'].append(self.loss)
        self.history['acc'].append(self.acc)
        self.tensorboard.add_scalar("loss", self.loss, self.rnd)
        self.tensorboard.add_scalar("accuracy", self.acc, self.rnd)
        print('rnd = 0, loss = {}, accuracy = {}'.format(self.loss, self.acc))
        # Clients' Operation
        old_weights = copy.deepcopy(self.aggregate_weights)

        # weights_lis = []  # Store qualified client weights for weighted average
        weights_dict = {}  # The serial number of the selected client and its corresponding weight

        for cid in self.cids:
            set_parameters(self.net, old_weights)
            num_pic, labels, steps = train_real(self.net, self.trainloaders[cid], epochs=1, lr=args.lr)
            new_weight = get_parameters(self.net)
            # weights_lis.append(new_weight)
            weights_dict[cid] = copy.deepcopy(new_weight)
        # Server's Operation
        score_cid = {}
        for cid in self.cids:
            # server computes gradient
            global_feature = 0
            weight_cid = weights_dict[cid]
            input_gradient = [torch.from_numpy((w2 - w1) / (args.lr * steps)).to(**setup) for w1, w2 in
                              zip(weight_cid, old_weights)]
            input_gradient = [grad.detach() for grad in input_gradient]

            # server learns the distribution
            set_parameters(self.net, self.aggregate_weights)
            self.net.eval()
            self.net.zero_grad()
            rec_machine = GradientReconstructor(self.net, (self.dm, self.ds), self.config,
                                                num_images=args.dummy_batch_size)  # args.dummy_batch_size)

            output, stats, recovered_labels = rec_machine.reconstruct(input_gradient, None,
                                                                      img_shape=self.image_shape)  # dummy_batch,
            feature = self.extract_feature(output).view(args.dummy_batch_size,
                                                        -1).detach()
            self.client_state[cid]['current_feature'] = (copy.deepcopy(feature), stats['opt'])
            global_feature += feature
        global_feature = global_feature / len(self.cids)
        for cid in self.cids:
            self.client_state[cid]['global_feature'] = copy.deepcopy(global_feature)
            sim_lc = 0.9
            sim_lg = 0.9
            sim_cg = maximum_mean_discrepancy(self.client_state[cid]['current_feature'][0].cpu(),
                                              global_feature.cpu())
            score_cid[cid] = [1-self.client_state[cid]['current_feature'][1],sim_lc, sim_lg, sim_cg]

            self.client_state[cid]['local_feature'] = copy.deepcopy(self.client_state[cid]['current_feature'])
            self.client_state[cid]['times'] += 1

        new_state = []
        for cid in self.cids:
            value = score_cid[cid]
            new_state.append(value)
        new_state = np.array(new_state)
        self.old_state = new_state
        self.weights_dict = weights_dict
        # ***********************************************************************************************

        return  new_state

import os
import copy
from utilities import *
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import csv
import torchvision.transforms as transforms
from PIL import Image

from utilities import _validate_config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RL_GradientReconstructor():
    """Instantiate a reconstruction algorithm."""


    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        #self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.iDLG = True

    def reconstruct(self, input_data, labels, DummySet, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape, DummySet)
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
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            if torch.numel(scores) == 0:
                stats['opt'] = 1000
                print("badbadbadbad")
                return _, stats, _
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]
            label_optimal = all_labels[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats, label_optimal

    def _init_images(self, img_shape, DummySet):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'pre':
            #print(torch.stack([DummySet for _ in range(self.config['restarts'])]).shape)
            #return torch.unsqueeze(DummySet,0)
            return torch.stack([DummySet for _ in range(self.config['restarts'])])
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            #labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)
            labels = torch.randn((self.num_images,output_test.shape[1])).to(**self.setup).requires_grad_(True)
            #print(labels.shape)
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

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
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

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')
                    elif iteration % 10 == 0 and self.config['optim'] == 'LBFGS':
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

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
            #print(label.shape)
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
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats
def craft(old_weights, new_weights, action, b, type="Clipping_Median"):
    # print(new_weights[-1][1])
    # zeta_max, zeta_min = b*0.0030664504, b*-0.0024578273
    # zeta_max=[zeta_layer*b for zeta_layer in zeta_max]
    # zeta_min=[zeta_layer*b for zeta_layer in zeta_min]
    weight_diff = [w1 - w2 for w1, w2 in zip(old_weights, new_weights)]  # weight_diff = grad*lr here
    crafted_weight_diff = [b * diff_layer * action for diff_layer in weight_diff]
    if type == "Clipping_Median":
        vec_weight_diff = weights_to_vector(crafted_weight_diff)
        crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    else:
        vec_weight_diff = weights_to_vector(crafted_weight_diff)
        crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    # crafted_weight_diff = [diff_layer* (action*(zeta_max-zeta_min)/abs(zeta_max)*0.5+(zeta_max+zeta_min)/abs(zeta_max)*0.5) for diff_layer in weight_diff]
    # crafted_weight_diff = [diff_layer* (action*(max_layer-min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5
    # +(max_layer+min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5)
    # for diff_layer, max_layer, min_layer in zip(weight_diff, zeta_max, zeta_min)]

    crafted_weight = [w1 - w2 for w1, w2 in zip(old_weights, crafted_weight_diff)]  # old_weight - lr*gradient
    return crafted_weight


def Median_craft_real(old_weights, weights_lis, att_ids, cids, net, agent_loaders, lr=0.05):
    """Craft Median weights."""

    temp_weights_lis = copy.deepcopy(weights_lis)
    for aid in att_ids:
        set_parameters(net, old_weights)
        train_real(net, agent_loaders[aid], epochs=1, lr=lr)
        # train(net, train_iter, epochs=1, lr=lr)
        new_weight = get_parameters(net)
        temp_weights_lis.append(new_weight)

    aggregate_weight = Median(old_weights, temp_weights_lis)
    # aggregate_weight = average(weights_lis)
    sign = [np.sign(u - v) for u, v in zip(aggregate_weight, old_weights)]
    # print(sign)

    max_weight = weights_to_vector(temp_weights_lis[0])
    min_weight = weights_to_vector(temp_weights_lis[0])
    for i in range(1, len(temp_weights_lis)):
        max_weight = np.maximum(max_weight, weights_to_vector(temp_weights_lis[i]))
        min_weight = np.minimum(min_weight, weights_to_vector(temp_weights_lis[i]))

    b = 5
    crafted_weights = []

    #    for _ in range(len(att_ids)):
    for _ in range(1):
        crafted_weight = []
        count = 0
        for layer in sign:

            new_parameters = []
            # print(layer.flatten())
            for parameter in layer.flatten():
                if parameter == -1. and max_weight[count] > 0:
                    new_parameters.append(random.uniform(max_weight[count], b * max_weight[count]))
                if parameter == -1. and max_weight[count] <= 0:
                    new_parameters.append(random.uniform(max_weight[count], max_weight[count] / b))
                if parameter == 1. and min_weight[count] > 0:
                    new_parameters.append(random.uniform(min_weight[count] / b, min_weight[count]))
                if parameter == 1. and min_weight[count] <= 0:
                    new_parameters.append(random.uniform(b * min_weight[count], min_weight[count]))
                if parameter == 0.: new_parameters.append(0)
                if np.isnan(parameter):
                    new_parameters.append(random.uniform(min_weight[count], max_weight[count]))
                count += 1
            # print(new_parameters)
            crafted_weight.append(np.array(new_parameters).reshape(layer.shape))
        # crafted_weights.append(crafted_weight)
    #crafted_weights = [crafted_weight for _ in range(len(att_ids))]
    return crafted_weight



def craft_att(old_weights, new_weights, action, b):
    #print(new_weights[-1][1])
    #zeta_max, zeta_min = b*0.0030664504, b*-0.0024578273
    #zeta_max=[zeta_layer*b for zeta_layer in zeta_max]
    #zeta_min=[zeta_layer*b for zeta_layer in zeta_min]
    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    vec_weight_diff = weights_to_vector(crafted_weight_diff)
    #print(np.linalg.norm(vec_weight_diff))
    crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    #crafted_weight_diff = [diff_layer* (action*(zeta_max-zeta_min)/abs(zeta_max)*0.5+(zeta_max+zeta_min)/abs(zeta_max)*0.5) for diff_layer in weight_diff]
    #crafted_weight_diff = [diff_layer* (action*(max_layer-min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5
                                        #+(max_layer+min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5)
                                        #for diff_layer, max_layer, min_layer in zip(weight_diff, zeta_max, zeta_min)]

    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight


def default_loader(path):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        #transforms.CenterCrop(370),
        #transforms.CenterCrop(280),
        transforms.Resize(28),
        #transforms.CenterCrop(28),
        transforms.ToTensor()
    ])
    img_pil =  Image.open(path)
    #print(img_pil)
    img_pil = img_pil.crop((143,58,512,426))
    img_tensor = preprocess(img_pil)
    return img_tensor

class Distribution_set(Dataset):
    def __init__(self, datapath='fashion_test/no_process', labelpath = 'fashion_test/data.csv', loader=default_loader):
        self.path = datapath
        files = os.listdir(self.path)
        num_png = len(files)
        self.images = []
        self.target = []
        self.loader = loader
        file = open(labelpath)
        reader = csv.reader(file)
        labels = []
        for row in reader:
            labels.append(row)
        print(len(labels))
        file.close()
        for i in range(num_png):
            if i % 1000 == 0:
                print("processing image ", i)
            fn = str(i)+'.png'
            img = self.loader(self.path+'/'+fn)
            self.target.append(int(labels[i][1]))
            self.images.append(img)

    def __getitem__(self, index):
        #fn = str(index)+'.png'
        #img = self.loader(self.path+'/'+fn)
        #target = int(self.target[index][1])
        return self.images[index],self.target[index]

    def __len__(self):
        return len(self.images)


def IPM_attack(net, old_weights, cids, att_ids):
    print("----------IPM Attack--------------")
    # for i in range(len(common(cids, att_ids))):
    #     set_parameters(net, old_weights)
    #     # train(net, train_iter, epochs=1, lr=lr)
    crafted_weights = {}
    for cid in common(cids, att_ids):
        crafted_weights[cid] = craft(old_weights, get_parameters(net), 5, -1)

    return crafted_weights


def LMP_attack(net, old_weights, cids, att_ids, trainloaders, weights_lis):
    print("----------LMP Attack--------------")
    crafted_weights = {}
    weight = Median_craft_real(old_weights, weights_lis, common(cids, att_ids), cids, net,
                      trainloaders)
    for cid in common(cids, att_ids):
        crafted_weights[cid] = weight
    return crafted_weights


def EB_attack(net, old_weights, cids, att_ids, trainloaders, testloaders):
    print("----------EB Attack--------------")
    print(len(common(cids, att_ids)))
    crafted_weights = {}
    for cid in common(cids, att_ids):
        set_parameters(net, old_weights)
        train_real_ga(net, trainloaders[cid], epochs=5, lr=0.05)
        loss, acc = test(net, testloaders[cid])
        new_weight = get_parameters(net)
        # print(self.rnd, loss, acc)
        check = 5
        while np.isnan(loss):
            check = max(check - 1, 0)
            set_parameters(net, old_weights)
            train_real(net, testloaders[cid], epochs=check, lr=0.05)
            new_weight = get_parameters(net)
            loss, acc = test(net, testloaders[cid])
            # print(rnd, loss, acc, check)
            if check == 0:
                new_weight = copy.deepcopy(old_weights)
                break
        crafted_weights[cid] = craft(old_weights, new_weight, 1, len(cids) / len(common(cids, att_ids)))
    return crafted_weights


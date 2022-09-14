"""Mechanisms for image reconstruction from parameter gradients."""

import torch
from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy

import time

DEFAULT_CONFIG = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='topk-1',
                      weights='equal',
                      lr=0.01,
                      optim='adam',
                      restarts=128,
                      max_iterations=8_000,
                      total_variation=0,
                      init='randn',
                      filter='none',
                      lr_decay=False,
                      scoring_choice='loss',
                      ratio=1.0,
                      username='none',
                      method ='none')

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor_attack():
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

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)
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
            for trial in range(self.config['restarts']):
                x_trial, labels, ls_all = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                
                
                
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels, ls_all)
                x[trial] = x_trial
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
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            ############################### over here ###############################

            ### my add-ons
            dropratio_str = self.config['ratio']
            dropratio = float(self.config['ratio'])
            #### in the case of dp, dropratio is the sigma value
            username =  self.config['username']
            method = self.config['method']
            username = username.replace("model", "grad")
            import os.path
            histgrad = torch.load(os.path.dirname(__file__) + '/../gg_contents/'+dropratio_str+"/"+username+'.pt')

            weights = input_data[0].new_ones(len(input_data))
            scorelist = []
            pnorm = [0.0, 0.0]
            costs = 0
            
            #print(histgrad.keys())
            counter = 0
            #print("histgrad length is {}".format(len(histgrad.keys())))
            for i in histgrad.keys():
                if "running_mean" in i or "running_var" in i or "num_batches_tracked" in i:
                    #print('skipped in {}'.format(i[10:]))
                    continue
                costs = (histgrad[i] * input_data[counter]).sum() * weights[counter]
                #print('costshape = {} {} {}'.format(((histgrad[i] * input_data[counter]).sum()).shape, costs.shape, weights[counter]))
                pnorm[0] = histgrad[i].pow(2).sum() * weights[counter]
                pnorm[1] = input_data[counter].pow(2).sum() * weights[counter]
                scorelist.append((counter,1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()))
                counter+=1

            #for global
            sortedscores = sorted(scorelist, key=lambda tup : tup[1], reverse=True) # sorted in decreasing score
            
            sortedlayers = []
            for pair_ in sortedscores:
                sortedlayers.append(pair_[0])

            layers_selected = sortedlayers[:int(len(input_data)*dropratio)]
            #print("layers selected length is {}".format(len(layers_selected)))
            
            ls_all = {}
            ls_all[self.config['username']] = layers_selected
            
            pairs = [("gg_model27", "airplane"), ("gg_model15", "automobile"), ("gg_model49", "bird"), ("gg_model49", "cat"), ("gg_model15", "deer"), ("gg_model51", "dog"), ("gg_model18", "frog"), ("gg_model15", "horse"), ("gg_model52", "ship"), ("gg_model15", "truck")]
            
            for pair in pairs:
                layers = []
                for model in ls_all.keys():
                    for num in range(len(ls_all[model])):
                        layers.append(ls_all[model][num])
                if len(layers) == len(input_data):
                    print("len of layers is {}".format(len(layers)))
                    print("broke")
                    break
                
                #get model and global gradient
                #calculate the gradient of image using above model
                modelname = pair[0]
                calcgrad = torch.load(os.path.dirname(__file__) + '/../attack_contents/'+modelname+'.pt')
                gradname = pair[0]
                gradname = gradname.replace("model", "grad")
                ggrad = torch.load(os.path.dirname(__file__) + '/../gg_contents/'+dropratio_str+"/"+username+'.pt')
                
                #find simlarity between the two gradients
                scorelist = []
                pnorm = [0.0, 0.0]
                costs = 0

                counter = 0
                skippednum = 0
                #print("keys(layers) is {} and length is {}".format(ggrad.keys(), len(ggrad.keys())))
                for i in ggrad.keys():
                    if "running_mean" in i or "running_var" in i or "num_batches_tracked" in i:
                        #print('skipped in {}'.format(i[10:]))
                        skippednum +=1
                        continue
                    costs = (ggrad[i] * calcgrad[counter]).sum()
                    #print('costshape = {} {} {}'.format(((histgrad[i] * input_data[counter]).sum()).shape, costs.shape, weights[counter]))
                    pnorm[0] = ggrad[i].pow(2).sum()
                    pnorm[1] = calcgrad[counter].pow(2).sum()
                    scorelist.append((counter,1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()))
                    counter += 1
                
                sortedscores = sorted(scorelist, key=lambda tup : tup[1], reverse=True) # sorted in decreasing score
                sortedlayers = []
                for pair_ in sortedscores:
                    sortedlayers.append(pair_[0])

                layers_selected = sortedlayers[:int(len(input_data)*dropratio)]
                #print("number of skipped layers is {}".format(skippednum))
                
                #take gradient layers idx of model and add to ls_all
                for layer in layers_selected:
                    if layer not in layers:
                        if modelname not in ls_all.keys():
                            ls_all[modelname] = []
                        ls_all[modelname].append(layer)
                    
            #alter input_data
            for model in ls_all.keys():
                #print(ls_all[model])
                
                if model == self.config['username']:
                    continue
                layers_selected = ls_all[model]

                calcgrad = torch.load(os.path.dirname(__file__) + '/../attack_contents/'+model+'.pt')
                
                
                for layer in layers_selected:
                    input_data[layer] = calcgrad[layer]
           
                
                
            
            
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels, ls_all)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if iteration % 100 == 0:   # or iteration + 1 == max_iterations:
                        print(f'It: {iteration + 1} out of {max_iterations}. Rec. loss: {rec_loss.item():2.4f}.')

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
        return x_trial.detach(), labels, ls_all

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, layers_selected):
    ########################################### here as well #####################################
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient, layers_selected,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])
                
            #alter gradient here
            
            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label, layers_selected):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient, layers_selected,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
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
        stats['opt'] = reconstruction_costs([gradient], input_data, layers_selected,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats



class FedAvgReconstructor(GradientReconstructor_attack):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for _ in range(local_steps):
        outputs = patched_model(inputs, patched_model.parameters)
        loss = loss_fn(outputs, labels).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, layers_selected, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is hijacked user gradient"""
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
    
    #print(len(gradients))
    for model in layers_selected.keys():
        for trial_gradient in gradients:
            pnorm = [0.0, 0.0]
            costs = 0
            if indices == 'topk-2':
                _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
            for i in layers_selected[model]:
                if cost_fn == 'l2':
                    costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
                elif cost_fn == 'l1':
                    costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
                elif cost_fn == 'max':
                    costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
                elif cost_fn == 'sim':
                    #print(layers_selected)
                    #print(input_gradient[i])
                    costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                    pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
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
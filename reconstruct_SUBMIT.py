import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import sys
import inversefed
import arrange


def run(arch, dataset, model_name, img_dir, img_class, gg_dir, method, layer_ratio):
    
    num_images = 1
    trained_model = True

    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    if dataset == "EMNIST":
        param1 = 'MNIST'
    else:
        param1 = dataset
        
    loss_fn, trainloader, validloader =  inversefed.construct_dataloaders(param1, defs)

    model, _ = inversefed.construct_model(arch, num_classes=arrange.dataset_size[dataset], num_channels=arrange.dataset_channel[dataset])
    model.to(**setup)
    if trained_model:
        print("    Using trained model")
        # file = f'gg_contents/'+layer_ratio+"/"+model_name+'.pt'
        # file = f'hg_contents_new/hg_model2.ckpt'
        file = nn_dir
        try:
            ckpt = torch.load(f'{file}')
            #model = ckpt
            model.load_state_dict(ckpt)
            #print(ckpt.keys())
            print("    Loaded correctly")
            

        except FileNotFoundError:
            print("uhoh")
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), f'models/{file}')
    model.eval();

    if dataset == "EMNIST":
        param2 = inversefed.consts.emnist_mean
        param3 = inversefed.consts.emnist_std
    elif dataset == "CIFAR10":
        param2 = inversefed.consts.cifar10_mean
        param3 = inversefed.consts.cifar10_std
    else:
        param2 = inversefed.consts.cifar100_mean
        param3 = inversefed.consts.cifar100_std
    
    dm = torch.as_tensor(param2, **setup)[:, None, None]
    ds = torch.as_tensor(param3, **setup)[:, None, None]

    def plot(tensor):
        tensor = tensor.clone().detach()
        tensor.mul_(ds).add_(dm).clamp_(0, 1)
        if tensor.shape[0] == 1:
            return plt.imshow(tensor[0].permute(1, 2, 0).cpu());
        else:
            fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
            for i, im in enumerate(tensor):
                axes[i].imshow(im.permute(1, 2, 0).cpu());


    # Reconstruct

    # Build the input (ground-truth) gradient

    img_name = img_dir.split(".")[0]
                   
    save_name = img_name+"_recon.png"
    outputfolder = "results/"
    
    if num_images == 1:
        if dataset =="EMNIST":
            ground_truth_image = torch.as_tensor(np.expand_dims(np.array(Image.open(img_dir).convert('L').resize((28, 28), Image.BICUBIC)), axis=2) / 255, **setup)
        else:
            ground_truth_image = torch.as_tensor(np.array(Image.open(img_dir).resize((32, 32), Image.BICUBIC)) / 255, 
                                             **setup)
        ground_truth = ground_truth_image.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        labels = torch.as_tensor((int(img_class),), device=setup['device'])
    else:
        ground_truth, labels = [], []
        idx = 25 # choosen randomly ... just whatever you want
        while len(labels) < num_images:
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))
        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)


    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    
    ###print(input_gradient[0].shape)

    ###print(len(input_gradient))
    
    """
    for i in range(len(input_gradient)):
        print(input_gradient[i].shape)
    """

    input_gradient = [grad.detach() for grad in input_gradient]


    # In[8]:


    #print(model)

    config = dict(signed=True,
                  boxed=True,
                  cost_fn='sim',
                  indices='def',
                  weights='equal',
                  lr=0.1,
                  optim='adam',
                  restarts=1,
                  max_iterations=4000,
                  total_variation=1e-6,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    
    if method == "global" or method == "dp" or method == "random":
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    elif method == "dgc":
        rec_machine = inversefed.GradientReconstructor_dgc(model, (dm, ds), config, num_images=num_images)
    else:
        print("wrong method name")
        sys.exit()

    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(arrange.dataset_channel[dataset], arrange.dataset_shape[dataset], arrange.dataset_shape[dataset]))
    
    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, ground_truth)
    
    """
    plot(output)
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
              f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");
    """
    with open(outputfolder+'/stats.txt', 'a+') as f:
        f.write(save_name+"    "+
              f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
              f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |\n")

    #save_image(output, 'reconstructimages/'+inputimage+"_stats.png")
    
    from torchvision.utils import save_image
    output_unnorm = output.mul(ds).add(dm)
    save_image(output_unnorm, outputfolder+"/"+save_name)

    
if __name__ == "__main__":
    
    arch = arrange.arch
    dataset = arrange.dataset
    nn_dir = arrange.nn_dir
    method = arrange.method
    gg_dir = arrange.gg_dir
    ratio = arrange.ratio
    
    
    img_dir = sys.argv[1]
    img_class = sys.argv[2]
    
    
    print("    {}: Reconstructing {} when layer ratio {}".format(method, img_dir, ratio))
    run(arch, dataset, nn_dir, img_dir, img_class, gg_dir, method, ratio)
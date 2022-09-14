README

Contains code used for ACSAC submission "Closing the Loophole: Rethinking Reconstruction Attacks in Federated Learning from a Privacy Standpoint".

FFL and its comparison defense methods (random selection, dp, dgc) are implemented on top of the reconstruction attack framework from https://github.com/JonasGeiping/invertinggradients.

Setup Requirements:
pytorch>=1.4.0
torchvision>=0.5.0

reconstruct_SUBMIT.py reconstructs an input image when applying a specific defense method.
The result will be saved in the results folder, along with the MSE and PSNR scores.

The model architecture, dataset, defense method, and layer selection ratio can be arranged in [arrange.py].
The dataset/architecture pairs that are available are the ones described in the text: (CIFAR10, ConvNet), (CIFAR10, ResNet), (CIFAR100, ResNet), (EMNIST, ConvNet).
The defense methods that can be used are "global" for FFL, "random" for random layer selection, "dp" for differential privacy, and "dgc" for deep gradient compression.
In the case of "dgc" the ratio refers to the compression rate.

data.tar.gz needs to be downloaded from from zenodo (https://zenodo.org/record/7079413#.YyHlZHZBwuU), due to its size (contains the required model weights).
After unpacking it so that there is a [data] folder in the working directory, the following can be done.

syntax: 
    python3 reconstruct_SUBMIT.py [image_path] [class_idx]

example:
First, to retrieve the outcome of a reconstruction attack when the model is defended by FFL (ratio 0.4) on the CIFAR10 dataset with architecture ConvNet, edit the arrange.py file like the following.
    #model
    arch = "ConvNet" 
    dataset = "CIFAR10"

    #defense
    method = "global" 
    ratio = "0.4"    
Then to reconstruct an image [cifar10_ex.png] that has the "airplane" label (index 0), the syntax becomes
    python3 reconstruct_SUBMIT.py cifar10_ex.png 0
The resulting image and its stats will be reported in the results folder.
dataset_size = {}
dataset_size["CIFAR10"]=10
dataset_size["CIFAR100"]=100
dataset_size["EMNIST"]=26
dataset_channel = {}
dataset_channel["CIFAR10"]=3
dataset_channel["CIFAR100"]=3
dataset_channel["EMNIST"]=1
dataset_shape = {}
dataset_shape["CIFAR10"]=32
dataset_shape["CIFAR100"]=32
dataset_shape["EMNIST"]=28

#model
arch = "ConvNet"          #  "ConvNet", "ResNet"
dataset = "EMNIST"      #  "CIFAR100", "CIFAR10", "EMNIST"

#defense
method = "global"        # "global", "dgc", "dp", "random"
ratio = "0.2"            # "0.2", "0.4", "0.6", "0.8", "1.0"

if method == "dp":
    nn_dir = "data/"+dataset+"_"+arch+"/dp_model.pt"
    gg_dir = "data/"+dataset+"_"+arch+"/dp_grad.pt"
else:
    nn_dir = "data/"+dataset+"_"+arch+"/gg_model.pt"
    gg_dir = "data/"+dataset+"_"+arch+"/gg_grad.pt"
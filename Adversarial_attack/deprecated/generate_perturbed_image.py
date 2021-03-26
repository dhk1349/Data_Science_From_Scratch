from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import attack_modules as attack
import dataloader
import SpinalNet
from ResNet import resnet50
import torchvision.models.resnet as resnet

if __name__ == '__main__':

    _attack = "deepfool"
    num_sample = 10
    use_cuda = True
    save_path = "/home/dhk1349/바탕화면/dataset/"

    print("Batch doesn't work now.")
    print(f"attack: {_attack}")
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    import torch
    #resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    resnet50 = resnet.resnet50(pretrained=True)
    resnet50.to(device)
    resnet50.eval()
    """
    resnet50 = resnet50()
    resnet50.load_state_dict(torch.load("./weight/state_dicts/resnet50.pt"))
    resnet50.to(device)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    """
    trainset = torchvision.datasets.CIFAR10(root="/home/dhk1349/바탕화면/dataset", train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="/home/dhk1349/바탕화면/dataset", train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    """
    valset = torchvision.datasets.ImageFolder(root="/media/dhk1349/Donghoon/dataset/ILSVRC2012_img_val", transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                             shuffle=True, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #print(resnet18)
    resnet50.eval()
    epsilons = [0, .05, .1, .15, .2, .25, .3, .4]

    accuracies = []
    examples = []

    if _attack == "fgsm":
        print("fgsm attack")
        for eps in epsilons:
            acc, ex = attack.attack_test(resnet50, device, valloader, epsilons)
            accuracies.append(acc)
            examples.append(ex)

        plt.figure(figsize=(5, 5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0.7, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
        # Plot several examples of adversarial samples at each epsilon
        cnt = 0
        plt.figure(figsize=(8, 10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons), len(examples[0]), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig, adv, ex = examples[i][j]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.show()

    elif _attack == "deepfool":
        print("deepfool attack")
        acc, examples = attack.attack_test(resnet50, device, valloader)

        print(f"saving data \n normal: {type(examples[:,3])}\n perturbed: {type(examples[:,2])}")
        #for _, _, nor, per in examples:

        np.save("./Imagenet_perturbed/deepfool/cifar10_deepfool_10.npy", examples)
        #np.save("./deepfool_samples/normal_samples", examples[:,3])
        #np.save("./deepfool_samples/perturbed_samples", examples[:,2])

        # Plot several examples of adversarial samples at each epsilon
        #cnt = 0
        #plt.figure(figsize=(1, 5))
        """
        for i in range(len(examples)):
            cnt += 1
            plt.subplot(1, 5, cnt)
            plt.xticks([], [])
            plt.yticks([], [])

            orig, adv, ex = examples[i]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.show()
        """
    print("exiting")


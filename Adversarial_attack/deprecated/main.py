from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import attack_modules as attack
import dataloader
import SpinalNet

if __name__ == '__main__':
    print("Batch doesn't work now.")
    _attack="deepfool"
    print(f"attack: {_attack}")
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    spinal = torch.load("./weight/pretrained_spinalnet.pt").to(device)

    print(spinal)
    spinal.eval()
    dl = dataloader.get_test_loader(path="/home/dhk1349/바탕화면/dataset", batch_size_test=1)
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    accuracies = []
    examples = []

    if _attack == "fgsm":
        print("fgsm attack")
        for eps in epsilons:
            acc, ex = attack.attack_test(spinal, device, dl)
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
        acc, examples = attack.attack_test(spinal, device, dl)

        print(f"saving data \n normal: {type(examples[:,3])}\n perturbed: {type(examples[:,2])}")
        #for _, _, nor, per in examples:

        np.save("./deepfool_samples/total", examples)
        np.save("./deepfool_samples/normal_samples", examples[:,3])
        np.save("./deepfool_samples/perturbed_samples", examples[:,2])

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


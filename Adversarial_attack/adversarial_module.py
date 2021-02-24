from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import numpy as np
import matplotlib.pyplot as plt

class attack_module():
    def __init__(self):
        self.name = ""

    def fgsm(self, image, target, net, epsilon=None):
        """

            :param image:
            :param target:
            :param net:
            :param epsilon:
            :return:


        """
        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the data through the model
        output = net(image)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.squeeze(-1).item() != target.item():
            return init_pred, image

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        net.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = image.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return init_pred, perturbed_image

    def deepfool(self, image, net, device='cpu', num_classes=10, overshoot=0.02, max_iter=10):
        """
        things to fix
        detach function
        var names

        :param image:
        :param net:
        :param device:
        :param num_classes:
        :param overshoot:
        :param max_iter:
        :return:
        """
        init_pred = 0
        # init_pred = output.max(1, keepdim=True)[1]
        f_image = net(image).data.cpu().numpy().flatten()  #예측

        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        label = I[0]  # switch to init_pred값

        # input_shape = image.detach().numpy().shape
        input_shape = image.size()
        perturbed_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        # [None, :]을 하면 None 남은숫자를 알아서 맞춰주게 된다.
        # 아래처럼 사용하게 되면 겉에 []를 한 겹 더 씌운다고 생각하면 된다.
        x = torch.tensor(perturbed_image[None, :], requires_grad=True).to(device=device)

        # 이럴거면 왜 감쌌지?
        fs = net(x[0])

        # fs_list는 fs를 I의 인덱스에 맞춰서(conf가 큰 라벨)순으로 conf를 나열한 듯 하다.
        # 개별적인 텐서로 저장함
        fs_list = [fs[0, I[k]] for k in range(num_classes)]

        k_i = label

        # label is initial prediction
        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):

                # x.zero_grad()

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy().copy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            perturbed_image = image + (1 + overshoot) * torch.from_numpy(r_tot).to(device=device)

            x = torch.tensor(perturbed_image, requires_grad=True)
            fs = net.forward(x[0])
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1 + overshoot) * r_tot

        # return r_tot, loop_i, label, k_i, perturbed_image
        return label, perturbed_image[0]

    def pgd(self):
        return
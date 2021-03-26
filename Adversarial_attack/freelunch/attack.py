import os
import numpy as np
import torch


class attack:

    def __init__(self):
        self.model
        self.batch

    def FGSM(self, image, target, net, epsilon=None):
        """
        :param image:
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
        return

    def PGD(self):
        return

    def deepfool(self):
        return

    def CW(self):
        return


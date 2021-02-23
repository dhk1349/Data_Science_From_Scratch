from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import numpy as np
import matplotlib.pyplot as plt


def fgsm(image, target, net, epsilon=None):
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


def deepfool(image, net, device='cpu', num_classes=10, overshoot=0.02, max_iter=10):
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
    f_image = net(image).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]  # switch to init_pred

    # input_shape = image.detach().numpy().shape
    input_shape = image.size()
    perturbed_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(perturbed_image[None, :], requires_grad=True).to(device=device)

    fs = net(x[0])

    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

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


def pgd(image, net, device='cpu', num_classes=10, overshoot=0.02, max_iter=10):

    return


def attack_test(model, device, test_loader, epsilon=None):
    """
    :explanation:


    :param model:
    :param device:
    :param test_loader:
    :param epsilon:
    :return:
    """

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        """
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.squeeze(-1).item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        """
        # Call FGSM Attack
        # init_pred, perturbed_data = fgsm(data, target, model, epsilon)
        init_pred, perturbed_data = deepfool(data, model, device)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))

    if epsilon is not None:
        print("Test Accuracy = {} / {} = {}".format(
            correct, len(test_loader), final_acc))
    else:
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

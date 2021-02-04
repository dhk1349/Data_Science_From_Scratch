import os
from datetime import date
from dataloader import *
from SpinalNet import *
import matplotlib.pyplot as plt

#Changed code from SpinalNet_MNIST.py for further use.
#https://github.com/dipuk0506/SpinalNet/blob/master/MNIST/SpinalNet_MNIST.py

if __name__ == "__main__":
    print(os.getcwd())

    n_epochs = 8

    torch.backends.cudnn.enabled = True

    train_loader = get_train_loader(path="/home/dhk1349/바탕화면/dataset")
    test_loader = get_test_loader(path="/home/dhk1349/바탕화면/dataset")
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_losses = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = train(network, train_loader, train_losses, train_counter, optimizer, epoch)
        test_losses=test(network, test_loader, test_losses)
    # need to store .pt file
    torch.save(network, "./weight/spinalnet_"+date.today().strftime('%m_%d_%Y')+'.pt')

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

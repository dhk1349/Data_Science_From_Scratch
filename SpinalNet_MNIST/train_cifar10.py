import os
import numpy as np
from datetime import date
from dataloader import *
from SpinalNet import *
from SpinalNet2 import *
import matplotlib.pyplot as plt

# Changed code from SpinalNet_MNIST.py for further use.
# https://github.com/dipuk0506/SpinalNet/blob/master/MNIST/SpinalNet_MNIST.py

if __name__ == "__main__":
    print(os.getcwd())

    n_epochs = 8

    torch.backends.cudnn.enabled = True

    train_loader = cifar10_loader(path="/media/dhk1349/second_disk/dataset/cifar10", train=True)
    test_loader = cifar10_loader(path="/media/dhk1349/second_disk/dataset/cifar10", train=False)
    classes = cifar10_get_class()

    plt.ion()  # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((272, 272)),
            transforms.RandomRotation(15, ),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
    }
    torchvision.datasets.CIFAR10
    data_dir = "/media/dhk1349/second_disk/dataset/cifar10"
    image_datasets = {idx: torchvision.datasets.CIFAR10(root=data_dir, train=x,
                                                        transform=data_transforms[idx])
                      for idx, x in {"train": True, "test": False}.items()}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=56,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    time_elapsed = time.time() - since
                    print('Time from Start {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))
    # 정답(label) 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    network = SpinalNet_ResNet().to("cuda:0")
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    network = train_model(network, criterion, optimizer_ft, exp_lr_scheduler,
                          num_epochs=5)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_losses = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = train(network, train_loader, train_losses, train_counter, optimizer, epoch)
        test_losses = test(network, test_loader, test_losses)
    # need to store .pt file
    torch.save(network, "./weight/spinalnet_" + date.today().strftime('%m_%d_%Y') + '.pt')

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

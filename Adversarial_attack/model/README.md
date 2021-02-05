# Spinal Network with MNIST dataset



### Changed code SpinalNet_MNIST.py for further use.
### https://github.com/dipuk0506/SpinalNet/blob/master/MNIST/SpinalNet_MNIST.py





:desktop_computer: training log below :desktop_computer:

/home/dhk1349/anaconda3/envs/torch/bin/python /home/dhk1349/바탕화면/github/Deep-Learning/SpinalNet_MNIST/main.py
/home/dhk1349/바탕화면/github/Deep-Learning/SpinalNet_MNIST
torch.Size([1000, 1, 28, 28])
/home/dhk1349/바탕화면/github/Deep-Learning/SpinalNet_MNIST/SpinalNet.py:71: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
/home/dhk1349/anaconda3/envs/torch/lib/python3.6/site-packages/torch/nn/_reduction.py:44: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Avg. loss: 2.3258, Accuracy: 938/10000 (9%)

Train Epoch: 1 [0/60000 (0%)]	Loss: 2.366093
Train Epoch: 1 [6400/60000 (11%)]	Loss: 2.019270
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.797680
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.384153
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.606186
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.367561
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.345898
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.271699
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.277888
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.238350

Test set: Avg. loss: 0.2200, Accuracy: 9330/10000 (93%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 0.342821
Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.343000
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.342953
Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.164569
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.421607
Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.086453
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.193541
Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.225601
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.247786
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.208087

Test set: Avg. loss: 0.1157, Accuracy: 9653/10000 (97%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 0.108567
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.195200
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.211482
Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.121020
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.151417
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.166892
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.129708
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.284024
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.040699
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.225524

Test set: Avg. loss: 0.0848, Accuracy: 9742/10000 (97%)

Train Epoch: 4 [0/60000 (0%)]	Loss: 0.295315
Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.068878
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.178153
Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.169596
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.324497
Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.119029
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.087267
Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.097631
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.078896
Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.105541

Test set: Avg. loss: 0.0798, Accuracy: 9735/10000 (97%)

Train Epoch: 5 [0/60000 (0%)]	Loss: 0.031337
Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.198479
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.334488
Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.117071
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.158114
Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.200506
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.250714
Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.039968
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.130057
Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.105440

Test set: Avg. loss: 0.0665, Accuracy: 9787/10000 (98%)

Train Epoch: 6 [0/60000 (0%)]	Loss: 0.220523
Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.069987
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.287752
Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.105060
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.033154
Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.230399
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.077272
Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.134537
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.013609
Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.026710

Test set: Avg. loss: 0.0600, Accuracy: 9802/10000 (98%)

Train Epoch: 7 [0/60000 (0%)]	Loss: 0.112843
Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.167933
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.199389
Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.068237
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.130952
Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.061988
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.083291
Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.205111
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.044809
Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.071345

Test set: Avg. loss: 0.0579, Accuracy: 9809/10000 (98%)

Train Epoch: 8 [0/60000 (0%)]	Loss: 0.123893
Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.084427
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.096382
Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.104978
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.144892
Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.039869
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.196685
Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.024100
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.132422
Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.019158

Test set: Avg. loss: 0.0554, Accuracy: 9815/10000 (98%)


Process finished with exit code 0
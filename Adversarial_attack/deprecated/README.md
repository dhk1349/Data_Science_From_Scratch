# Adversarial Attack with MNIST 

I used pre-trained network from 

https://github.com/dhk1349/Deep-Learning/tree/master/SpinalNet_MNIST



> FGSM

Test accuracy without any perturbation was approximately 97%.

I tried "Fast Gradient Sign Method"(FGSM) attack on MNIST test dataset with given model.

Test accuracy dropped a little as epsilon value increases. However, accuracy did not drop drastically. I'm not sure why acc did not drop much but to give some headcanon, I think it's 

because model was too accurate and dataset was too easy.



If dataset had more various labels to classify, i guess accuracy would drop more rapidly.



> Deepfool

Test accuracy with Deepfool attack on MNIST scored 0.3355% (pre-trained Spinal Network). 

However, it's no wonder that discriminator scores so low with this attack method because deepfool attacks image iteratively.

Since deepfool attacks image iteratively, it is much stronger than FGSM attack (needless to say)



:memo:  execution log



> FGSM

```
CUDA Available:  True
Net(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=160, out_features=8, bias=True)
  (fc1_1): Linear(in_features=168, out_features=8, bias=True)
  (fc1_2): Linear(in_features=168, out_features=8, bias=True)
  (fc1_3): Linear(in_features=168, out_features=8, bias=True)
  (fc1_4): Linear(in_features=168, out_features=8, bias=True)
  (fc1_5): Linear(in_features=168, out_features=8, bias=True)
  (fc2): Linear(in_features=48, out_features=10, bias=True)
)

Epsilon: 0	Test Accuracy = 9705 / 10000 = 0.9705
Epsilon: 0.05	Test Accuracy = 9676 / 10000 = 0.9676
Epsilon: 0.1	Test Accuracy = 9641 / 10000 = 0.9641
Epsilon: 0.15	Test Accuracy = 9611 / 10000 = 0.9611
Epsilon: 0.2	Test Accuracy = 9572 / 10000 = 0.9572
Epsilon: 0.25	Test Accuracy = 9510 / 10000 = 0.951
Epsilon: 0.3	Test Accuracy = 9452 / 10000 = 0.9452

```



> Deepfool

```
Batch doesn't work now.
attack: deepfool
CUDA Available:  True
Net(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=160, out_features=8, bias=True)
  (fc1_1): Linear(in_features=168, out_features=8, bias=True)
  (fc1_2): Linear(in_features=168, out_features=8, bias=True)
  (fc1_3): Linear(in_features=168, out_features=8, bias=True)
  (fc1_4): Linear(in_features=168, out_features=8, bias=True)
  (fc1_5): Linear(in_features=168, out_features=8, bias=True)
  (fc2): Linear(in_features=48, out_features=10, bias=True)
)
deepfool attack
Epsilon: None	Test Accuracy = 3355 / 10000 = 0.3355
exiting

Process finished with exit code 0
```



:heavy_plus_sign: planning to add more attack methods

:heavy_plus_sign: ​planning to change attack_modules into class structure so that it is mode usable.
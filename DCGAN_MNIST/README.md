# DCGAN with MNIST dataset



This code is mostly from pytorch tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html



I changed dataset using MNIST dataset in DCGAN_MNIST.

Generator seems to generate some meaningful outcomes after training. Generated outputs were recognizable. However, generated outputs looked somewhat clumsy.



Also there is a code block that uses specific number as training data. (it's commented out in the fourth block of code)



One thing to emphasize

if options in Discriminator and Generator (e.g. kernel size, step size..)  change, performance of networks drop drastically. So when you are thinking of implementing different dataset, padding the size of dataset would be a better option.




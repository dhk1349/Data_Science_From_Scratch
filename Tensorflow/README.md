Purpose of this repo is to learn how to use Tensorflow







### +jupyter on virtualenv(Anaconda)

-Install two libraries below.

pip install ipykernel

pip install jupyter

Then, type

python -m ipykernel install --user --name "username"

![virtualenv](C:\Users\한동훈\Desktop\virtualenv.PNG)

Like this, jupyer notebook is open and login as tf_env(virtual env)



### +tf.gpu/CUDA

I tried to link GPU with tensorflow using CUDA.

![tf_gpu](C:\Users\한동훈\Desktop\tf_gpu.PNG)



Before instlling CUDA, first  step should be checking compatible version among python, tensorflow-gpu, CUDA, and cuDNN.

![gpu_comp](C:\Users\한동훈\Desktop\gpu_comp.PNG)

If you download incompatible version, tensorflow may not link CUDA.

My recommandation for installation step is below.

1. Choose which tf-gpu version to install.

2. Install CUDA and cuDNN.

3. After installing CUDA, unzip cuDNN and copy&paste cuDNN files to CUDA folder. 

   ​	a) CUDA folder is typically located on usr/Program Files/NVIDIA GPU Computing Toolkit

   ​	b) If you want to make sure CUDA is sucessfully installed, 

   ​		type "nvcc --version" and "nvidia-smi"

4. Make sure virtualenv has compatible python version.

5. Install tensorflow-gpu (check versions): pip install tensorflow-gpu==2.0.0

6. import tensorflow as tf

7. tf.test.is_gpu_available()

8. function above will return True if GPU is available on tensorflow


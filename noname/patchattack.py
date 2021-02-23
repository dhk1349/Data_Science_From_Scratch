import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


print("hello world")


noise=np.random.randn(256,256,3)
#print(noise)
image = Image.fromarray(noise, "RGB")
image.show()
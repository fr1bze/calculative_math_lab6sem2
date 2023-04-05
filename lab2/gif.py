import numpy as np
import matplotlib.pyplot as plt 
import pylab as plb
from PIL import Image

k = 16
arr = np.arange(1,k)

def save2Gif(range):
    images=[]
    image = Image.open("plot_0.png")
    for i in range:
        if i!=0:
            name = 'plot_'+ str(i)+'.png'
            images.append(Image.open(name))
    image.save('pic.gif', save_all=True, append_images=images,loop=100,duration=1)
save2Gif(arr)

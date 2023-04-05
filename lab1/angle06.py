import numpy as np
import matplotlib.pyplot as plt 
import pylab as plb
from PIL import Image

#consts
cfl = 0.6
T = 18.0
L = 20.0
h = 0.5


t1 = cfl * h
x = np.array([(i-1)*h for i in range(1,int(L/h +1)+1)])
u0 = np.sin(6*np.pi*x/L)
u = u0.copy()
plt.plot(x,u,'y')
plb.xlim([0,10])
plb.ylim([-1,1])
plb.grid()
plt.savefig('image0.jpg')
plb.close()
plb.figure()
cnt = 1
rng = T/t1
for i in range(int(rng)):
  un = u.copy()
  for i in range(1,len(x)):
    u[i]=un[i]-(t1/h)*(un[i]-un[i-1])
    #print(u[i])
  u[0]=u[len(x)-1]
  plb.close()
  plb.plot(x,u,'y')
  plb.xlim([0,10])
  plb.ylim([-1,1])
  plb.grid()
  plb.savefig('image'+str(cnt)+'.jpg')
  cnt+=1


arr_of_images = np.arange(1,cnt)
def save2Gif(rang):
    im = Image.open("image0.jpg")
    images=[]
    for i in rang:
        if i!=0:
            fpath = 'image'+str(i)+'.jpg'
            images.append(Image.open(fpath))
    im.save('УголокCFL06.gif', save_all=True, append_images=images,loop=100,duration=1)
save2Gif(arr_of_images)
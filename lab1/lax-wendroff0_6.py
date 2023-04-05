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
plt.plot(x,u,'blue')
plb.xlim([0,10])
plb.ylim([-1,1])
plb.grid()
plt.savefig('image__0.jpg')
plb.close()
plb.figure()
cnt = 1
rng = T/t1
for i in range(int(rng)):
  un = u.copy()
  for i in range(1,len(x)-1):
    u[i]=un[i]-(t1/(2*h))*(un[i+1]-un[i-1])+(t1**2/(2*h*h))*(un[i+1]-2*un[i]+un[i-1])
    #bordary conditions
  u[len(un)-1]=un[len(un)-1]-(t1/(2*h))*(un[0]-un[len(un)-2])+(t1**2/(2*h*h))*(un[0]-2*un[len(un)-1]+un[len(un)-2])
  u[0]=un[0]-(t1/(2*h))*(un[1]-un[len(un)-1])+(t1**2/(2*h*h))*(un[1]-2*un[0]+un[len(un)-1])
  plb.close()
  plb.plot(x,u,'blue')
  plb.xlim([0,10])
  plb.ylim([-1,1])
  plb.grid()
  plb.savefig('image__'+str(cnt)+'.jpg')
  cnt+=1


arr_of_images = np.arange(1,cnt)
def save2Gif(rang):
    im = Image.open("image__0.jpg")
    images=[]
    for i in rang:
        if i!=0:
            fpath = 'image__'+str(i)+'.jpg'
            images.append(Image.open(fpath))
    im.save('Lax-Wendroff06.gif', save_all=True, append_images=images,loop=100,duration=1)
save2Gif(arr_of_images)
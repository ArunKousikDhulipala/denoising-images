#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[20]:


#1 5 10 30 100
def diffusion_kernel(d,t):
    x,y=np.meshgrid(np.linspace(-27,27,55),np.linspace(-27,27,55))
    dst = x**2+y**2
    nrml=1/((4*d*t*np.pi)**0.5)
    exponent=np.exp(-dst/(4*d*t))
    gauss=exponent*nrml
    k=np.sum(gauss)
    kernel=gauss/k
    
    return kernel


kernel_constd_t1=diffusion_kernel(1,1)
kernel_constd_t5=diffusion_kernel(1,5)
kernel_constd_t10=diffusion_kernel(1,10)
kernel_constd_t30=diffusion_kernel(1,30)
kernel_constd_t100=diffusion_kernel(1,100)
kernel_constd_t1000=diffusion_kernel(1,1000)

#kernel_constd_t100


# In[21]:


noisy_img=cv2.imread("C:\\Users\\User\\Downloads\\office_noisy.png")


# In[22]:


img_1=cv2.filter2D(noisy_img,-1,kernel_constd_t1)
img_2=cv2.filter2D(noisy_img,-1,kernel_constd_t5)
img_3=cv2.filter2D(noisy_img,-1,kernel_constd_t10)
img_4=cv2.filter2D(noisy_img,-1,kernel_constd_t30)
img_5=cv2.filter2D(noisy_img,-1,kernel_constd_t100)

img_6=cv2.filter2D(noisy_img,-1,kernel_constd_t1000)


# In[23]:


plt.imshow(noisy_img)
plt.title("Noisy image")
plt.axis("off")
plt.show


# In[24]:


k=img_1
plt.imshow(k)
plt.title("t=1")
plt.axis("off")
plt.show


# In[25]:


plt.imshow(img_2)
plt.title("t=5")
plt.axis("off")
plt.show


# In[26]:


plt.imshow(img_3)
plt.title("t=10")
plt.axis("off")
plt.show


# In[27]:


plt.imshow(img_4)
plt.title("t=30")
plt.axis("off")
plt.show


# In[28]:


plt.imshow(img_5)
plt.title("t=100")
plt.axis("off")
plt.show


# In[29]:


k1000=img_6
plt.imshow(k1000)
plt.title("t=1000")
plt.axis("off")
plt.show


# In[30]:


#display the image with isotropic diffusion : t= 1 and d=1 kernel size 55x55
plt.subplot(2,3,1)
plt.imshow(img_1)
plt.title("t=1 and d=1")
plt.axis("off")

#display the image with isotropic diffusion : t= 5 and d=1 kernel size 55x55
plt.subplot(2,3,2)
plt.imshow(img_2)
plt.title("t=5 and d=1")
plt.axis("off")

#display the image with isotropic diffusion : t= 10 and d=1 kernel size 55x55
plt.subplot(2,3,3)
plt.imshow(img_3)
plt.title("t=10 and d=1")
plt.axis("off")


#display the image with isotropic diffusion : t= 30 and d=1 kernel size 55x55
plt.subplot(2,3,4)
plt.imshow(img_4)
plt.title("t=30 and d=1")
plt.axis("off")


#display the image with isotropic diffusion : t= 100 and d=1 kernel size 55x55
plt.subplot(2,3,5)
plt.imshow(img_5)
plt.title("t=100 and d=1")
plt.axis("off")




plt.show()


# In[31]:


kernel_constt_d1=diffusion_kernel(1,10)
kernel_constt_d5=diffusion_kernel(5,10)
kernel_constt_d10=diffusion_kernel(10,10)


# In[32]:


dimg_1=cv2.filter2D(noisy_img,-1,kernel_constt_d1)
dimg_2=cv2.filter2D(noisy_img,-1,kernel_constt_d5)
dimg_3=cv2.filter2D(noisy_img,-1,kernel_constt_d10)


# In[33]:


plt.imshow(noisy_img)
plt.title("Noisy image")
plt.axis("off")
plt.show


# In[34]:


plt.imshow(dimg_1)
plt.title("d=1 , t=10")
plt.axis("off")
plt.show


# In[35]:


plt.imshow(dimg_2)
plt.title("d=5, t=10")
plt.axis("off")
plt.show


# In[36]:


plt.imshow(dimg_3)
plt.title("d=10, t=10")
plt.axis("off")
plt.show


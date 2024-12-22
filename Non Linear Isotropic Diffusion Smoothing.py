#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("C:\\Users\\User\\Downloads\\office_noisy.png")
img.shape


# In[5]:


def filter_(img,time_step,itr,lambd):
    
    for i in range(itr):
        gradient_x=cv2.Sobel(img,cv2.CV_64F,1,0,9)
        gradient_y=cv2.Sobel(img,cv2.CV_64F,0,1,9)
        gradient=(np.sqrt(gradient_x**2 + gradient_y**2))
        diffusivity=1/(1+(gradient/lambd)**2)
        img=img+time_step*(cv2.Sobel(gradient_x*diffusivity,cv2.CV_64F,1,0,9)+cv2.Sobel(gradient_y*diffusivity,cv2.CV_64F,0,1,9))
        img=np.clip(img,0,255)

        
    return img.astype("uint8")


# In[6]:


k=filter_(img,0.2,500,0.5)
plt.imshow(k)
plt.show


# In[7]:


k1=filter_(img,0.2,5,0.5) #t=1 lambda=0.5
k5=filter_(img,0.2,25,0.5) #t=5 lambda=0.5
k10=filter_(img,0.2,50,0.5) #t=10 lambda=0.5
k30=filter_(img,0.2,150,0.5) #t=30 lambda=0.5
k100=filter_(img,0.2,500,0.5) #t=100 lambda=0.5


# In[8]:


#display the noisy image 
plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Noisy img")
plt.axis("off")

#display the image with t=1 lambda=0.5 
plt.subplot(2,3,2)
plt.imshow(k1)
plt.title("t=1 , lambda=0.5")
plt.axis("off")

#display the image with t=5 lambda=0.5 
plt.subplot(2,3,3)
plt.imshow(k5)
plt.title("t=5 , lambda=0.5")
plt.axis("off")

#display the image with t=10 lambda=0.5 
plt.subplot(2,3,4)
plt.imshow(k10)
plt.title("t=10 , lambda=0.5")
plt.axis("off")

#display the image with t=30 lambda=0.5
plt.subplot(2,3,5)
plt.imshow(k30)
plt.title("t=30 , lambda=0.5")
plt.axis("off")

#display the image with t=100 lambda=0.5
plt.subplot(2,3,6)
plt.imshow(k100)
plt.title("t=100 , lambda=0.5")
plt.axis("off")

plt.show


# In[9]:


limg=cv2.imread("C:\\Users\\User\\Downloads\\lena256.png")


# In[10]:


l1=filter_(limg,0.2,5,0.5) #t=1 lambda=0.5
l5=filter_(limg,0.2,25,0.5) #t=5 lambda=0.5
l10=filter_(limg,0.2,50,0.5) #t=10 lambda=0.5
l30=filter_(limg,0.2,150,0.5) #t=30 lambda=0.5
l100=filter_(limg,0.2,500,0.5) #t=100 lambda=0.5


# In[11]:


#display the noisy image 
plt.subplot(2,3,1)
plt.imshow(limg)
plt.title("Normal img")
plt.axis("off")

#display the image with t=1 lambda=0.5 
plt.subplot(2,3,2)
plt.imshow(l1)
plt.title("t=1 , lambda=0.5")
plt.axis("off")

#display the image with t=5 lambda=0.5 
plt.subplot(2,3,3)
plt.imshow(l5)
plt.title("t=5 , lambda=0.5")
plt.axis("off")

#display the image with t=10 lambda=0.5 
plt.subplot(2,3,4)
plt.imshow(l10)
plt.title("t=10 , lambda=0.5")
plt.axis("off")

#display the image with t=30 lambda=0.5
plt.subplot(2,3,5)
plt.imshow(l30)
plt.title("t=30 , lambda=0.5")
plt.axis("off")

#display the image with t=100 lambda=0.5
plt.subplot(2,3,6)
plt.imshow(l100)
plt.title("t=100 , lambda=0.5")
plt.axis("off")

plt.show


# In[12]:


plt.imshow(l1)
plt.axis("off")
plt.show


# In[13]:


plt.imshow(l100)
plt.axis("off")
plt.show


# In[14]:


lmd05=filter_(limg,0.2,50,0.5) #t=10 lambda=0.5
lmd1=filter_(limg,0.2,50,1) #t=10 lambda=1
lmd2=filter_(limg,0.2,50,2) #t=10 lambda=2
lmd5=filter_(limg,0.2,50,5) #t=10 lambda=5
lmd10=filter_(limg,0.2,50,10) #t=10 lambda=10


# In[15]:


#display the noisy image 
plt.subplot(2,3,1)
plt.imshow(limg)
plt.title("Normal img")
plt.axis("off")

#display the image with t=10 lambda=0.5 
plt.subplot(2,3,2)
plt.imshow(lmd05)
plt.title("t=10 , lambda=0.5")
plt.axis("off")

#display the image with t=10 lambda=1 
plt.subplot(2,3,3)
plt.imshow(lmd1)
plt.title("t=10 , lambda=1")
plt.axis("off")

#display the image with t=10 lambda=2 
plt.subplot(2,3,4)
plt.imshow(lmd2)
plt.title("t=10 , lambda=2")
plt.axis("off")

#display the image with t=10 lambda=5
plt.subplot(2,3,5)
plt.imshow(lmd5)
plt.title("t=10 , lambda=5")
plt.axis("off")

#display the image with t=10 lambda=10
plt.subplot(2,3,6)
plt.imshow(lmd10)
plt.title("t=10 , lambda=10")
plt.axis("off")

plt.show


# In[16]:


lmd_05=filter_(img,0.2,50,0.5) #t=10 lambda=0.5
lmd_1=filter_(img,0.2,50,1) #t=10 lambda=1
lmd_2=filter_(img,0.2,50,2) #t=10 lambda=2
lmd_5=filter_(img,0.2,50,5) #t=10 lambda=5
lmd_10=filter_(img,0.2,50,10) #t=10 lambda=10


# In[17]:


#display the noisy image 
plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Normal img")
plt.axis("off")

#display the image with t=10 lambda=0.5 
plt.subplot(2,3,2)
plt.imshow(lmd_05)
plt.title("t=10 , lambda=0.5")
plt.axis("off")

#display the image with t=10 lambda=1 
plt.subplot(2,3,3)
plt.imshow(lmd_1)
plt.title("t=10 , lambda=1")
plt.axis("off")

#display the image with t=10 lambda=2 
plt.subplot(2,3,4)
plt.imshow(lmd_2)
plt.title("t=10 , lambda=2")
plt.axis("off")

#display the image with t=10 lambda=5
plt.subplot(2,3,5)
plt.imshow(lmd_5)
plt.title("t=10 , lambda=5")
plt.axis("off")

#display the image with t=10 lambda=10
plt.subplot(2,3,6)
plt.imshow(lmd_10)
plt.title("t=10 , lambda=10")
plt.axis("off")

plt.show


# In[18]:


plt.imshow(lmd_05)
plt.axis("off")
plt.show


# In[19]:


plt.imshow(lmd_10)
plt.axis("off")
plt.show


# In[20]:


office=cv2.imread("C:\\Users\\User\\Downloads\\office.png")
grad_x=cv2.Sobel(office,cv2.CV_64F,1,0,5) # utilizing the sobel function to calculate the Gradient of image wrt x 
grad_y=cv2.Sobel(office,cv2.CV_64F,0,1,5) # utilizing the sobel function to calculate the Gradient of image wrt y 
lamda=0.5
grad=np.sqrt(grad_x**2 + grad_y**2) # magnitude of the gradient of the image
diffusion_function=1/(1+(grad/lamda)**2) # calculating the diffusion function

# code for the diffusion function


# In[28]:


plt.imshow(diffusion_function)
plt.title("D(x,y) function")
plt.axis("off")
plt.show


# In[24]:


l1000=filter_(img,0.2,5000,0.5) #t=100 lambda=0.5


# In[26]:


plt.imshow(l1000)
plt.title("t=1000,lambda=0.5")
plt.axis("off")
plt.show


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary packages
import cv2
from matplotlib import pyplot as plt


# In[2]:


#loading the noisy image
noisy_img=cv2.imread("C:\\Users\\User\\Downloads\\office_noisy.png")


# In[3]:


#the image before denoising
plt.imshow(noisy_img)
plt.axis("off")
plt.show()


# In[4]:


res_img1=cv2.GaussianBlur(noisy_img,(55,55),0.5) #Gaussian Blur with sigma 0.5 and kernel size 55x55
res_img2=cv2.GaussianBlur(noisy_img,(55,55),1) #Gaussian Blur with sigma 1 and kernel size 55x55
res_img3=cv2.GaussianBlur(noisy_img,(55,55),2) #Gaussian Blur with sigma 2 and kernel size 55x55
res_img4=cv2.GaussianBlur(noisy_img,(55,55),5)#Gaussian Blur with sigma 5 and kernel size 55x55
res_img5=cv2.GaussianBlur(noisy_img,(55,55),10)#Gaussian Blur with sigma 10 and kernel size 55x55
res_img6=cv2.GaussianBlur(noisy_img,(55,55),50)#Gaussian Blur with sigma 50 and kernel size 55x55


# In[5]:


#display the image with Gaussian Blur : sigma 0.5 
plt.subplot(2,3,1)
plt.imshow(res_img1)
plt.title("sigma=0.5")
plt.axis("off")

#display the image with Gaussian Blur : sigma 1 
plt.subplot(2,3,2)
plt.imshow(res_img2)
plt.title("sigma=1")
plt.axis("off")

#display the image with Gaussian Blur : sigma 2 
plt.subplot(2,3,3)
plt.imshow(res_img3)
plt.title("sigma=2")
plt.axis("off")

#display the image with Gaussian Blur : sigma 5 
plt.subplot(2,3,4)
plt.imshow(res_img4)
plt.title("sigma=5")
plt.axis("off")

#display the image with Gaussian Blur : sigma 10
plt.subplot(2,3,5)
plt.imshow(res_img5)
plt.title("sigma=10")
plt.axis("off")

#display the image with Gaussian Blur : sigma 50
plt.subplot(2,3,6)
plt.imshow(res_img6)
plt.title("sigma=50")
plt.axis("off")

plt.show


# In[6]:


#display the image with no Gaussian Blur
plt.imshow(noisy_img)
plt.title("no smoothing")
plt.axis("off")
plt.show


# In[7]:


#display the image with Gaussian Blur : sigma 0.5 and kernel size 5x5
plt.imshow(res_img1)
plt.title("sigma=0.5")
plt.axis("off")
plt.show


# In[8]:


#display the image with Gaussian Blur : sigma 1 and kernel size 5x5
plt.imshow(res_img2)
plt.title("sigma=1")
plt.axis("off")
plt.show


# In[9]:


#display the image with Gaussian Blur : sigma 2 and kernel size 5x5
plt.imshow(res_img3)
plt.title("sigma=2")
plt.axis("off")
plt.show


# In[10]:


#display the image with Gaussian Blur : sigma 5 and kernel size 5x5
plt.imshow(res_img4)
plt.title("sigma=5")
plt.axis("off")
plt.show


# In[11]:


#display the image with Gaussian Blur : sigma 10 and kernel size 5x5
plt.imshow(res_img5)
plt.title("sigma=10")
plt.axis("off")
plt.show


# In[12]:


#display the image with Gaussian Blur : sigma 50 and kernel size 5x5
plt.imshow(res_img6)
plt.title("sigma=50")
plt.axis("off")
plt.show


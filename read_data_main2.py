# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:41:37 2024

@author: SAYAN GHOSH
"""

import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import cv2
import glob
import numpy as np
from PIL import Image
import os
from os import listdir


base_folder_path = "C:/Users/sinha/Desktop/epileptic_OCNN/sub5/Data1"

img3=[]

for i in range(0, 414):  # Change 11 to the number of folders you want to loop through
    print(i)
    folder_name = f"trial_{i}"
    folder_path = os.path.join(base_folder_path, folder_name)
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        #print(f"Folder '{folder_path}' exists.")
        img1=[]
        
        for j in range(64):  
            #print(j)
            image_name = f"topomap_sample_{j}.png"  # Assuming the images are JPG format
            image_path = os.path.join(folder_path, image_name)
            #print(image_path)
            img = cv2.imread(image_path) 
            #plt.imshow(img)
            # crop = img[100:300, 280:550,] 
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            target_size = (50, 50)  # Adjust the dimensions as needed
            resized_img = cv2.resize(gray_image, target_size)
            resized_img2=resized_img.astype('float32')/255.0
            img1.append(resized_img2)
        img3.append(img1)
            
img2=np.array(img1)
print(img2.shape)

img4=np.array(img3)
print(img4.shape)
            
np.save("img4_save", img4)
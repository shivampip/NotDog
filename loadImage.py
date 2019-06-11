import numpy as np
import pandas as pd
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from matplotlib import pyplot as plt 
from matplotlib import image as pp
import math

class LoadImages:
    
    
    def __init__(self, folderPath):
        self.folderPath = folderPath
        
    def getFiles(self):
        fileWithPaths=[]
        folder=self.folderPath
        fileNames=os.listdir(folder)
        for file in fileNames:
            if os.path.isfile(os.path.join(folder, file)):
                fileWithPaths.append(folder+"/"+file)
    
        return fileWithPaths
    
    def plotRawImage(self,numberofimages=4):
        files=self.getFiles()
        sampling = random.choices(files, k=numberofimages)
        n=math.ceil(math.sqrt(numberofimages))
        fig=plt.figure(figsize=(n*2,n*2))
        
        for index in range(1,len(sampling)+1):
            image=pp.imread(sampling[index-1])
            fig.add_subplot(n,n,index)
            plt.axis('off')
            plt.imshow(image)
        plt.show()    
    
    def plotNumpyImage(self,dataset,numberofimages=4):
        sampling=np.random.randint(dataset.shape[0],size=numberofimages)
        n=math.ceil(math.sqrt(numberofimages))
        fig=plt.figure(figsize=(n*2,n*2))
        
        for index in range(1,len(sampling)+1):
            image=dataset[sampling[index-1]]
            fig.add_subplot(n,n,index)
            plt.axis('off')
            #image=image.reshape(image.shape[1],image.shape[2],image.shape[0])
            plt.imshow(image)
        plt.show()    
    
    
    def ImageToNumpyArray(self,image_width,image_height,channels):
        folderPath=self.folderPath
        self.image_width=image_width
        self.image_height=image_height
        self.channels=channels
        filesWithPath=self.getFiles()
        dataset = np.ndarray(shape=(len(filesWithPath),image_height, image_width,channels),
                         dtype=np.int32)
        i=0
        for file in filesWithPath:
            img=load_img(file)
            img=img.resize(([image_height,image_width]))
            x = img_to_array(img)
            x = x.reshape((image_height,image_width,3))
            dataset[i] = x
            i += 1
        return dataset
          
	
# This class provides full image data importing and preprocessing pipeline.
# It makes data ready to feed any NN.

import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import math
import numpy as np 

class ImgData:
    
    # folder_path= folder containing images
    # out_dim= 128 x 128 pixel (like this)
    def __init__(self, folder_path):
        self.folder_path= folder_path
        self.out= []

   

    def get_all_files(self):
        self.files= os.listdir(self.folder_path) 
        self.count= len(self.files)
        self.files_path= [os.path.join(self.folder_path, fname) for fname in self.files]
        print("Total {} image files found".format(self.count))



    def plot_sample(self, rows=4, cols= 4):
        pix= self.files_path[0:rows*cols]
        
        fig= plt.figure(figsize= (cols*2, rows*2))
        
        for index in range(1, rows*cols +1):
            img= mpimg.imread(pix[index-1])
            fig.add_subplot(rows, cols, index) 
            plt.axis('off')
            plt.imshow(img) 
        plt.show()


    def plot_processed_sample(self, count= 4):
        if(count>len(self.out)):
            count= len(self.out)
        n= math.ceil(math.sqrt(count))
        fig= plt.figure(figsize= (n*2, n*2))
        for index in range(1, count+1):
            img= self.out[index-1]
            fig.add_subplot(n, n, index) 
            plt.axis('off')
            plt.imshow(img) 
        plt.show()


    def process(self, size= (150,150), count= -1):
        if(count==-1):
            self.n= self.count
        else:
            self.n= count 

        self.resize(size= size)
        #self.to_numpy()

        


    def resize(self, size=(150,150)):
        for path in self.files_path[:self.n]:
            print("Resizing {}".format(path))
            img= load_img(path, target_size= size)
            self.out.append(img)

    
    
    def to_numpy(self):
        nps= []
        for index in range(self.n):
            print("Converting {}".format(index))
            nps.append(img_to_array(self.out[index]))
        return nps
        
        

    

    

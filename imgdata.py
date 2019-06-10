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



    def plot_raw_sample(self, rows=4, cols= 4):
        pix= self.files_path[0:rows*cols]
        
        fig= plt.figure(figsize= (cols*2, rows*2))
        
        for index in range(1, rows*cols +1):
            img= mpimg.imread(pix[index-1])
            fig.add_subplot(rows, cols, index) 
            plt.axis('off')
            plt.imshow(img) 
        plt.show()


    def process(self, size= (150,150), count= -1, silent=False):
        if(count==-1):
            self.n= self.count
        else:
            self.n= count 

        self.resize(size= size, silent=silent)
        return self.to_numpy(silent=silent)

        


    def resize(self, size=(150,150), silent=False):
        print("Resizing", end= "")
        for path in self.files_path[:self.n]:
            if(not silent):
                print(".", end="")
            img= load_img(path, target_size= size)
            self.out.append(img)
        print(" DONE")

    
    
    def to_numpy(self, silent=False):
        nps= []
        print("Converting to NP", end="")
        for index in range(self.n):
            if(not silent):
                print(".", end="")
            nps.append(img_to_array(self.out[index]))
        print(" DONE")
        #return np.array([a for a in nps], dtype= np.int32)
        return np.array([a for a in nps])
        
    

    def plot_np_imgs(self, arr, count= 4):
        n= math.ceil(math.sqrt(count))
        fig= plt.figure(figsize= (n*2, n*2))
        for index in range(1, count+1):
            img= arr[index-1]
            fig.add_subplot(n, n, index) 
            plt.axis('off')
            plt.imshow(img) 
        plt.show()

        
    def shuffle(self, data):
        X, y= data[0]
        for i in range(1,len(data)):
            XX, yy= data[i]
            X= np.concatenate([X, XX], axis= 0)
            y= np.concatenate([y, yy], axis= 0)
        p= np.random.permutation(len(y))
        return (X[p],y[p])

    

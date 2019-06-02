# This class provides full image data importing and preprocessing pipeline.
# It makes data ready to feed any NN.

class ImgData:
    
    # folder_path= folder containing images
    # out_dim= 128 x 128 pixel (like this)
    def __init__(self, folder_path, out_dim):
        self.folder_path= folder_path


    # Do all kind of processing
    def process(self):
        self.resize()
        self.convert()
        #
        #
        #
        

    def resize(self):
        pass 

    def convert(self):
        pass 


    # Display images in (row, column)
    # (3,4) = Display total 12 ramdom images (3 rows, 4 columns)
    def show_sample(self, shape=(3,3)):
        pass
        
    

    
